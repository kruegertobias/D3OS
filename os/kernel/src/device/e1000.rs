use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use log::{info, warn};
use spin::{Mutex, RwLock};
use core::ops::BitOr;
use core::slice;
use bitflags::bitflags;
use nolock::queues::mpmc;
use x86_64::{structures::paging::{page::PageRange, Page, PageTableFlags}, VirtAddr};
use pci_types::{CommandRegister, EndpointHeader};
use smoltcp::phy;
use smoltcp::phy::{DeviceCapabilities, Medium};
use smoltcp::time::Instant;
use smoltcp::wire::EthernetAddress;
use crate::memory::vma::VmaType;
use crate::device::e1000_register;
use crate::device::e1000_register::E1000Register;
use crate::device::rtl8139::Interrupt;
use crate::interrupt::interrupt_dispatcher::InterruptVector;
use crate::{apic, interrupt_dispatcher, pci_bus};
use crate::interrupt::interrupt_handler::InterruptHandler;
use crate::process_manager;
use crate::memory::PAGE_SIZE;
use crate::memory::vmm::VirtualAddressSpace;
use crate::process::process;
use crate::syscall::sys_concurrent::sys_thread_sleep;

const DESCRIPTOR_BUFFER_SIZE: usize = 2048;
const NR_OF_DESCRIPTORS: usize = 256;
const RECV_QUEUE_CAP: usize = 64;

pub struct E1000 {
    mac: [u8; 6],
    mmio_virt_addr: u64,
    registers: E1000Register,
    rx_ring: RxRing,
    tx_ring: TxRing,
    rx_next: usize,
    interrupt: InterruptVector,
    recv_messages: (mpmc::bounded::scq::Receiver<Vec<u8>>, mpmc::bounded::scq::Sender<Vec<u8>>),
}


pub struct TxRing {
    pub vaddr: u64,
    pub paddr: u64,
    pub count: usize,
    pub len_bytes: usize
}

pub struct RxRing {
    pub vaddr: u64,
    pub paddr: u64,
    pub count: usize,
    pub len_bytes: usize,
}

#[repr(C, packed)]
pub struct TxDesc {
    pub buffer_addr: u64, // Physische Adresse des Sendepuffers
    pub length:      u16, // Länge des Pakets
    pub cso:         u8,
    pub cmd:         u8,  // EOP, IFCS, RS, ...
    pub status:      u8,  // Wird von HW gesetzt
    pub css:         u8,
    pub special:     u16,
}

#[repr(C, packed)]
pub struct RxDesc {
    pub buffer_addr: u64,
    pub length: u16,
    pub checksum: u16,
    pub status: u8,
    pub errors: u8,
    pub special: u16,
}
const RX_DESC_SIZE: usize = core::mem::size_of::<RxDesc>();
const TX_DESC_SIZE: usize = core::mem::size_of::<TxDesc>();

bitflags! {
    pub struct InterruptCause: u32 {
        const TXDW = 0x00000001;
        const TXQE = 0x00000002;
        const LSC = 0x00000004;
        const RXDMT0 = 0x00000010;
        const RXO = 0x00000040;
        const RXDW = 0x00000080;
    }
}

pub struct E1000InterruptHandler {
    device: Arc<E1000>,
}

impl E1000 {
    pub fn new(pci_device: &RwLock<EndpointHeader>) -> Self {
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

        let interrupt = InterruptVector::try_from(pci_device.interrupt(pci_config_space).1 + 32).unwrap();

        // map to virtual memory
        let kernel_process = process_manager().read().kernel_process().expect("No kernel process found!");
        let vmm = &kernel_process.virtual_address_space;
        let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::NO_CACHE;
        let start_page = vmm.kernel_map_devm_identity(base_address as u64, (base_address + size) as u64 , flags, VmaType::DeviceMemory, "e1000_mmio");
        let mmio_virt_addr = start_page.start_address().as_u64();
        info!("E1000 MMIO mapped at virtual address: {:#x}", mmio_virt_addr);

        let mut e1000register = E1000Register::new(mmio_virt_addr);

        let ctrl = e1000register.read_ctrl();
        log::info!("E1000 CTRL: {:#010x}", ctrl);

        let status = unsafe { e1000register.read_status() };
        log::info!("E1000 STATUS: {:#010x}", status);

        //set reset bit CTRL.RST(26)
        e1000register.write_ctrl(ctrl | (1 << 26));
        //Flush
        e1000register.read_ctrl();
        while e1000register.read_ctrl() & (1 << 26) != 0 {}

        udelay(5000);

        // read MAC Address from EEPROM
        let mac = read_mac_address(mmio_virt_addr);
        assert!(is_valid_mac(&mac));
        log::info!("MAC = {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

        init_ctrl(&mut e1000register);

        let rx_ring = init_receive_ring();
        init_receive_register(&mut e1000register, &rx_ring);

        let tx_ring = init_transmit_ring();
        init_transmit_register(&mut e1000register, &tx_ring);

        init_interrupt_registers(&e1000register);

        Self {
            mac,
            mmio_virt_addr,
            registers: e1000register,
            rx_ring,
            tx_ring,
            rx_next: 0,
            interrupt,
            recv_messages: mpmc::bounded::scq::queue(RECV_QUEUE_CAP),
        }
    }

    fn send_data(&self, data: *const u8, size: u32, eop: bool) {
        if size as usize > DESCRIPTOR_BUFFER_SIZE {
            panic!("E1000 send_data size exceeds descriptor buffer");
        }
        //load tail descriptor
        let tail = self.registers.read_tdt() as usize;
        let count = self.tx_ring.count;
        let base = self.tx_ring.vaddr as *mut TxDesc;
        let tx_desc = unsafe { &mut *base.add(tail % count) };

        let buffer_ptr = tx_desc.buffer_addr as *mut u8;
        unsafe {
            core::ptr::copy_nonoverlapping(data, buffer_ptr, size as usize);
        }

        //fill Descriptor
        tx_desc.length = size as u16;
        tx_desc.cmd = 0;
        tx_desc.status = 0;
        if eop {
            tx_desc.cmd |= (1 << 0) | (1 << 1);
        }

        //increase tail descriptor
        let next_tail = (tail + 1) % self.tx_ring.count;
        self.registers.write_tdt(next_tail as u32);
    }

    pub fn send(&self, data: *const u8, length: usize) -> usize {
        let mut sent = 0usize;

        while sent < length {
            let to_send = core::cmp::min(length - sent, DESCRIPTOR_BUFFER_SIZE);
            let chunk_ptr = unsafe { data.add(sent) };
            self.send_data(chunk_ptr, to_send as u32, to_send == (length - sent));
            sent += to_send;
        }
        info!("Data send!");
        sent
    }

    pub fn receive_packets(&self) -> Vec<Vec<u8>> {
        let mut idx = self.rx_next; //aktuell zu verarbeitender deskriptor
        let mut packets = Vec::new(); // zum speichern der pakete, ersetze mit Netzwerkstack bald
        let mut buffer: Vec<u8> = Vec::new();  //

        loop {
            let base = self.rx_ring.vaddr as *mut RxDesc;
            let rx_desc = unsafe { &mut *base.add(idx) };            if (rx_desc.status & (1 << 0)) == 0 { //ist der deskriptor gefühlt?
                break;
            }

            let eop = (rx_desc.status & (1 << 1)) != 0;
            let len = rx_desc.length as usize;
            let data_ptr = rx_desc.buffer_addr as *const u8;
            let data = unsafe { slice::from_raw_parts(data_ptr, len) };

            if buffer.is_empty() { //erster deskriptor des paketes?
                buffer = Vec::with_capacity(len);
            }
            buffer.extend_from_slice(data);

            rx_desc.status = 0;
            idx = (idx + 1) % self.rx_ring.count;

            if eop {
                //Letzter Deskriptor des Paketes, push das paket zum netzwerkstack
                info!("Received Packet");
                if self.recv_messages.1.try_enqueue(buffer).is_err() {
                    warn!("E1000 receive queue full, dropping packet");
                }
                buffer = Vec::new();
            }
        }

        // setze rdt auf den letzten bearbeiteten Deskriptor
        let tail = if idx == 0 {
            self.rx_ring.count - 1
        } else {
            idx - 1
        };
        self.registers.write_rdt(tail as u32);
        let mut ptr = &self.rx_next as *const usize as *mut usize;
        unsafe{ *ptr = idx; };

        packets
    }




    pub fn test_send(&self) {
        for _ in 0..2 {
            let payload = [0xFFu8; 2048];
            let sent = self.send(payload.as_ptr(), payload.len());
            info!("E1000 test_send: queued {} bytes", sent);
        }
    }

    pub fn plugin(device: Arc<E1000>) {
        let interrupt = device.interrupt;
        interrupt_dispatcher().assign(interrupt, Box::new(E1000InterruptHandler::new(device)));
        apic().allow(interrupt);
    }


    pub fn mac_address(&self) -> EthernetAddress {
        EthernetAddress::from_bytes(&self.mac)
    }
}

impl E1000InterruptHandler {
    pub fn new(device: Arc<E1000>) -> Self {
        Self { device }
    }
}

impl InterruptHandler for E1000InterruptHandler {
    fn trigger(&self) {
            info!("E1000 interrupt handler triggered");
            let status = InterruptCause::from_bits_retain(self.device.registers.read_icr());
            if status.intersects(InterruptCause::RXDW | InterruptCause::RXO | InterruptCause::RXDMT0) {
                self.device.receive_packets();
            }
    }
}

pub struct E1000TxToken<'a> {
    device: &'a E1000,
}

pub struct E1000RxToken {
    buffer: Vec<u8>,
}

impl<'a> phy::TxToken for E1000TxToken<'a> {
    fn consume<R, F>(self, len: usize, f: F) -> R
    where
        F: FnOnce(&mut [u8]) -> R,
    {
        let mut buffer = vec![0u8; len];
        let result = f(&mut buffer);

        self.device.send(buffer.as_ptr(), buffer.len());

        result
    }
}

impl phy::RxToken for E1000RxToken {
    fn consume<R, F>(self, f: F) -> R
    where
        F: FnOnce(&[u8]) -> R,
    {
        f(&self.buffer)
    }
}

impl phy::Device for E1000 {
    type RxToken<'a> = E1000RxToken where Self: 'a;
    type TxToken<'a> = E1000TxToken<'a> where Self: 'a;

    fn receive(&mut self, _timestamp: Instant) -> Option<(Self::RxToken<'_>, Self::TxToken<'_>)> {
        let recv_buf = self.recv_messages.0.try_dequeue().ok()?;
        Some((E1000RxToken { buffer: recv_buf }, E1000TxToken { device: self }))
    }

    fn transmit(&mut self, _timestamp: Instant) -> Option<Self::TxToken<'_>> {
        Some(E1000TxToken { device: self })
    }

    fn capabilities(&self) -> DeviceCapabilities {
        let mut caps = DeviceCapabilities::default();
        caps.max_transmission_unit = 1536;
        caps.max_burst_size = Some(1);
        caps.medium = Medium::Ethernet;
        caps
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

pub fn init_interrupt_registers(e1000register: &E1000Register) {
    e1000register.write_imc(u32::MAX);
    e1000register.read_icr();
    let mask = (InterruptCause::RXDW | InterruptCause::RXO | InterruptCause::RXDMT0).bits();
    e1000register.write_ims(mask);
}

pub fn is_valid_mac(mac: &[u8; 6]) -> bool {
    mac != &[0, 0, 0, 0, 0, 0] &&
        mac != &[0xFF; 6] &&
        (mac[0] & 1) == 0
}

pub fn init_ctrl(e1000register: &mut E1000Register) {
    let mut ctrl = e1000register.read_ctrl();
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


    e1000register.write_ctrl(ctrl);
    // clear FCAL FCAH FCT FCTTV
    e1000register.write_fcal(0x00000000);
    e1000register.write_fcah(0x00000000);
    e1000register.write_fct(0x00000000);
    e1000register.write_fcttv(0x00000000);

}

pub fn init_receive_ring() -> RxRing {
    // Alloc receive ringbuffer and set to 0
    let pages_ringbuffer = ((NR_OF_DESCRIPTORS * RX_DESC_SIZE) + PAGE_SIZE - 1) / PAGE_SIZE;
    let kernel_process = process_manager().read().kernel_process().expect("No kernel process found!");
    let vmm = &kernel_process.virtual_address_space;
    let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::NO_EXECUTE;
    let ring_rx_range = vmm.kernel_alloc_map_identity(pages_ringbuffer as u64, flags, VmaType::KernelBuffer, "rx_ringbuffer");
    let start = ring_rx_range.start.start_address().as_u64();
    let base_addr = start as *mut RxDesc;
    unsafe {
        core::ptr::write_bytes(
            base_addr as *mut u8,
            0,
            NR_OF_DESCRIPTORS * RX_DESC_SIZE,
        );
    }

    // alloc buffer for every descriptor
    let pages_buffer = (NR_OF_DESCRIPTORS * DESCRIPTOR_BUFFER_SIZE + PAGE_SIZE - 1) / PAGE_SIZE ;
    let flags_buf = PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::NO_EXECUTE;
    let buf_range = vmm.kernel_alloc_map_identity(
        pages_buffer as u64,
        flags_buf,
        VmaType::KernelBuffer,
        "rx_buffers",
    );
    let buf_base_paddr = buf_range.start.start_address().as_u64();

    //write every address in descriptor.buffer_addr
    for i in 0..NR_OF_DESCRIPTORS {
        let buf_paddr = buf_base_paddr + (i as u64) * (DESCRIPTOR_BUFFER_SIZE as u64);

        unsafe {
            let d = base_addr.add(i);

            (*d).buffer_addr = buf_paddr;
        }
    }

    RxRing {
        vaddr: start, // Need to fix - funktioniert nur durch 1 zu 1 phys-virt
        paddr: start,
        count: NR_OF_DESCRIPTORS,
        len_bytes: NR_OF_DESCRIPTORS * RX_DESC_SIZE
    }
}

pub fn init_receive_register(e1000register: &mut E1000Register, rx_ring: &RxRing) {
    let paddr = rx_ring.paddr;

    let rdbal = (paddr & 0xFFFF_FFFF) as u32;
    let rdbah = (paddr >> 32) as u32;

    let rdlen = rx_ring.len_bytes as u32;
    assert_eq!((rdlen & 0x7F), 0, "RDLEN must be 128-byte aligned");


    // 1) Descriptor Ring Base + Length
    e1000register.write_rdbal(rdbal);
    e1000register.write_rdbah(rdbah);
    e1000register.write_rdlen(rdlen);

    // 2) Head / Tail initialisieren
    e1000register.write_rdh(0);
    e1000register.write_rdt((rx_ring.count - 1) as u32);

    // 3) Receive Control Register
    let rctl =
        (1 << 1) |          // Receiver enable
            (1 << 5) |         // Long Packet Enable
                (1 << 15) |         // Broadcast Accept Mode
                    (1 << 6) |   // Loop Back Mode
                        (0b00 << 16);  // 2048 Byte Buffers (BSEX = 0)
    e1000register.write_rctl(rctl);
    // Flush
    e1000register.read_rctl();
}
pub fn init_transmit_ring() -> TxRing {
    // alloc for transmit ringbuffer
    let pages_ringbuffer = ((NR_OF_DESCRIPTORS * TX_DESC_SIZE) + PAGE_SIZE - 1) / PAGE_SIZE;
    info!("Pages Ringbuffer = {}", pages_ringbuffer);
    let kernel_process = process_manager().read().kernel_process().expect("No kernel process found!");
    let vmm = &kernel_process.virtual_address_space;
    let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::NO_EXECUTE;
    let ring_tx_range = vmm.kernel_alloc_map_identity(pages_ringbuffer as u64, flags, VmaType::KernelBuffer, "tx_ringbuffer");
    let start = ring_tx_range.start.start_address().as_u64();
    let base_addr = start as *mut TxDesc;
    unsafe {
        core::ptr::write_bytes(
            base_addr as *mut u8,
            0,
            NR_OF_DESCRIPTORS * TX_DESC_SIZE,
        );
    }
    // alloc buffer for descriptors
    let pages_buffers =  ((NR_OF_DESCRIPTORS * DESCRIPTOR_BUFFER_SIZE) + PAGE_SIZE - 1) / PAGE_SIZE;
    info!("Pages TX Buffers = {}", pages_buffers);
    let flags_buf = PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::NO_EXECUTE;
    let buf_range = vmm.kernel_alloc_map_identity(
        pages_buffers as u64,
        flags_buf,
        VmaType::KernelBuffer,
        "tx_buffers",
    );
    let buf_base_paddr = buf_range.start.start_address().as_u64();

    // add pointer to every descriptor
    for i in 0..NR_OF_DESCRIPTORS {
        let buf_paddr = buf_base_paddr + (i as u64) * (DESCRIPTOR_BUFFER_SIZE as u64);

        unsafe {
            let d = base_addr.add(i);

            (*d).buffer_addr = buf_paddr;
        }
    }

    TxRing {
        vaddr: start, // Need to fix
        paddr: start,
        count: NR_OF_DESCRIPTORS,
        len_bytes: NR_OF_DESCRIPTORS * TX_DESC_SIZE
    }
}

pub fn init_transmit_register(e1000register: &mut E1000Register, tx: &TxRing) {
    // 1) Descriptor Ring Base/Len
    let paddr = tx.paddr;
    let tdbal = (paddr & 0xFFFF_FFFF) as u32;
    let tdbah = (paddr >> 32) as u32;

    // TDLEN muss 128-Byte aligned sein (untere 7 Bits = 0)
    let tdlen = tx.len_bytes as u32;
    assert!((tdlen & 0x7F) == 0, "TDLEN must be 128-byte aligned");

    e1000register.write_tdbal(tdbal);
    e1000register.write_tdbah(tdbah);
    e1000register.write_tdlen(tdlen);

    // 2) Head/Tail initialisieren (nach Reset, vor EN)
    e1000register.write_tdh(0);
    e1000register.write_tdt(0);

    let tctl = e1000register.read_tctl();
    info!("E1000 TCTL: {:#010x}", tctl );

    // 4) TCTL: EN=1, PSP=1
    let tctl_val =
        (1u32 << 1) |          // EN
            (1u32 << 3);          // PSP
    e1000register.write_tctl(tctl_val);
    info!("E1000 TCTL: {:#010x}", tctl_val );

    //Flush
    e1000register.read_tctl();
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
