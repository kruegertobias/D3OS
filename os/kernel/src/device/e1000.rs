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



pub struct E1000 {
    // Fügen Sie hier die benötigten Felder hinzu
}

impl E1000 {
    pub fn new(pci_device: &RwLock<EndpointHeader>){
        info!("Configuring PCI registers");
        let pci_config_space = pci_bus().config_space();
        let mut pci_device = pci_device.write();
    
        // Make sure bus master and memory space are enabled for MMIO register access
        pci_device.update_command(pci_config_space, |command| {
            command.bitor(CommandRegister::BUS_MASTER_ENABLE | CommandRegister::MEMORY_ENABLE)
        });
        let register = pci_device.command(pci_bus().config_space());
        info!("Register value: {:?}", register);

        let bar0 = pci_device.bar(0, pci_bus().config_space()).expect("Failed to read base address!");
        info!("BAR0: {:?}", bar0);
        let (base_address, size) = bar0.unwrap_mem(); 
        info!("E1000 MMIO Base Address: {:#x}, Size: {:#x}", base_address, size);
        info!("End Phys_adrr: {:#x}", base_address+size);
        
        let kernel_process = process_manager().read().kernel_process().expect("No kernel process found!");
        let vmm = &kernel_process.virtual_address_space;
        let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::NO_CACHE;
        let start_page = vmm.kernel_map_devm_identity(base_address as u64, (base_address + size) as u64 , flags, VmaType::DeviceMemory, "e1000_mmio");
        let mmio_virt_addr = start_page.start_address().as_u64();
        info!("E1000 MMIO mapped at virtual address: {:#x}", mmio_virt_addr);

    }

    // Fügen Sie hier weitere Methoden hinzu
}