use log::info;
use spin::RwLock;
use core::ops::BitOr;



use pci_types::{CommandRegister, EndpointHeader};
use crate::pci_bus;




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
    
    }

    // Fügen Sie hier weitere Methoden hinzu
}