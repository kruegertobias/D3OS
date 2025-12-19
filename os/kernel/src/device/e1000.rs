use log::info;
use spin::RwLock;
use core::ops::BitOr;




use x86_64::{structures::paging::{page::PageRange, Page, PageTableFlags}, VirtAddr};
use pci_types::{CommandRegister, EndpointHeader};
use crate::memory::vma::VmaType;

use crate::pci_bus;
use crate::process_manager;
use crate::memory::PAGE_SIZE;
use crate::process::process;
use crate::syscall::sys_concurrent::sys_thread_sleep;

const CTRL: u32 = 0x00000;
const STATUS: u32 = 0x00008;
const FCAL: u32 = 0x00028;
const FCAH: u32 = 0x0002C;
const FCT: u32 = 0x00030;
const FCTTV: u32 = 0x00170;
const EERD: u32 = 0x00014;

pub struct E1000 {

}

impl E1000 {
    pub fn new(pci_device: &RwLock<EndpointHeader>){
        info!("Configuring PCI registers");
        let pci_config_space = pci_bus().config_space();
        let mut pci_device = pci_device.write();
    
        // Enable bus master and memory space for MMIO register access
        pci_device.update_command(pci_config_space, |command| {
            command.bitor(CommandRegister::BUS_MASTER_ENABLE | CommandRegister::MEMORY_ENABLE)
        });

        // read mmio base adress from bar0
        let bar0 = pci_device.bar(0, pci_bus().config_space()).expect("Failed to read base address!");
        let (base_address, size) = bar0.unwrap_mem(); 
        info!("E1000 MMIO Base Address: {:#x}, Size: {:#x}", base_address, size);

        // map to virtual memory
        let kernel_process = process_manager().read().kernel_process().expect("No kernel process found!");
        let vmm = &kernel_process.virtual_address_space;
        let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::NO_CACHE;
        let start_page = vmm.kernel_map_devm_identity(base_address as u64, (base_address + size) as u64 , flags, VmaType::DeviceMemory, "e1000_mmio");
        let mmio_virt_addr = start_page.start_address().as_u64();
        info!("E1000 MMIO mapped at virtual address: {:#x}", mmio_virt_addr);

        let ctrl = unsafe { mmio_read32(mmio_virt_addr, CTRL) };
        log::info!("E1000 CTRL: {:#010x}", ctrl);

        let status = unsafe { mmio_read32(mmio_virt_addr, STATUS) };
        log::info!("E1000 STATUS: {:#010x}", status);

        // set reset bit CTRL.RST(26)
        unsafe { mmio_write32(mmio_virt_addr, CTRL, ctrl | (1 << 26)) };
        //Flush
        unsafe { mmio_read32(mmio_virt_addr, CTRL) };
        udelay(5000);

        // read MAC Adress from EEPROM
        let mac = read_mac_address(mmio_virt_addr);
        assert!(is_valid_mac(&mac));
        log::info!("MAC = {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);



        init_ctrl(mmio_virt_addr);



    }

}
unsafe fn mmio_read32(base: u64, offset: u32) -> u32 {
    let addr = (base + offset as u64) as *const u32;
    core::ptr::read_volatile(addr)
}

unsafe fn mmio_write32(base: u64, offset: u32, value: u32) {
    let addr = (base + offset as u64) as *mut u32;
    core::ptr::write_volatile(addr, value);
}

pub fn udelay(us: usize) {
    for _ in 0..(us * 1000) {
        core::hint::spin_loop();
    }
}

pub fn eeprom_read(mmio_base: u64, addr: u8) -> u16 {
    // START=1 + Adresse setzen
    let cmd = ((addr as u32) << 8) | (1 << 0);
    unsafe {
        mmio_write32(mmio_base, 0x000014, cmd);
    }

    // Polling, bis DONE=1
    loop {
        let val = unsafe { mmio_read32(mmio_base, 0x00014) };
        if (val & (1<<4)) != 0 {
            return ((val >> 16) & 0xFFFF) as u16;
        }
    }
}

pub fn read_mac_address(mmio_base: u64) -> [u8; 6] {
    let w0 = eeprom_read(mmio_base, 0x00);
    let w1 = eeprom_read(mmio_base, 0x01);
    let w2 = eeprom_read(mmio_base, 0x02);

    [
        (w0 & 0x00FF) as u8,
        (w0 >> 8) as u8,
        (w1 & 0x00FF) as u8,
        (w1 >> 8) as u8,
        (w2 & 0x00FF) as u8,
        (w2 >> 8) as u8,
    ]
}


pub fn is_valid_mac(mac: &[u8; 6]) -> bool {
    mac != &[0, 0, 0, 0, 0, 0] &&
        mac != &[0xFF; 6] &&
        (mac[0] & 1) == 0
}

pub fn init_ctrl(mmio_virt_addr: u64) {
    let mut ctrl = unsafe { mmio_read32(mmio_virt_addr, CTRL) };
    // CRTL.ASDE CRTL.SLU
    ctrl = ctrl | (1 << 5) | (1 << 6);
    //clear CTRL.LRST
    ctrl = ctrl & !(1 << 3);
    //clear CTRL.PHY_RST
    ctrl = ctrl & !(1 << 31);
    //clear CTRL.ILOS
    ctrl = ctrl & !(1 << 7);
    //enable Receive Flow Control and Transmit Flow Control via Phy-Auto-Negotiation.
    ctrl |= (1 << 27) | (1 << 28);
    // clear CTRL.FRCSDP
    ctrl = ctrl & !(1 << 11);
    // clear FRCDPLX
    ctrl = ctrl & !(1 << 12);
    info!("E1000 CTRL: {:#010x}", ctrl);

    unsafe { mmio_write32(mmio_virt_addr, CTRL, ctrl) };

    // clear FCAL FCAH FCT FCTTV
    unsafe { mmio_write32(mmio_virt_addr, FCAL, 0x00000000) };
    unsafe { mmio_write32(mmio_virt_addr, FCAH, 0x00000000) };
    unsafe { mmio_write32(mmio_virt_addr, FCT, 0x00000000) };
    unsafe { mmio_write32(mmio_virt_addr, FCTTV, 0x00000000) };
}
