pub struct E1000Register {
    ctrl: u64,
    status: u64,
    fcal: u64,
    fcah: u64,
    fct: u64,
    fcttv: u64,
    eerd: u64,
    tdbal: u64,
    tdbah: u64,
    tdlen: u64,
    tdh: u64,
    tdt: u64,
    tctl: u64,
    tipg: u64,
    rdbal: u64,
    rdbah: u64,
    rdlen: u64,
    rdh: u64,
    rdt: u64,
    rctl: u64
}


impl E1000Register {
    pub fn new(mmio_virt_addr: u64) -> E1000Register {
        E1000Register {
            ctrl: mmio_virt_addr + 0x00000,
            status: mmio_virt_addr + 0x00008,
            fcal: mmio_virt_addr + 0x00028,
            fcah: mmio_virt_addr + 0x0002C,
            fct: mmio_virt_addr + 0x00030,
            fcttv: mmio_virt_addr + 0x00170,
            eerd: mmio_virt_addr + 0x00014,
            tdbal: mmio_virt_addr + 0x03800,
            tdbah: mmio_virt_addr + 0x03804,
            tdlen: mmio_virt_addr + 0x03808,
            tdh: mmio_virt_addr + 0x03810,
            tdt: mmio_virt_addr + 0x03818,
            tctl: mmio_virt_addr + 0x00400,
            tipg: mmio_virt_addr + 0x00410,
            rdbal: mmio_virt_addr + 0x02800,
            rdbah: mmio_virt_addr + 0x02804,
            rdlen: mmio_virt_addr + 0x02808,
            rdh: mmio_virt_addr + 0x02810,
            rdt: mmio_virt_addr + 0x02818,
            rctl: mmio_virt_addr + 0x00100
        }
    }

    // CTRL
    pub fn read_ctrl(&self) -> u32 {
        unsafe {
            core::ptr::read_volatile(self.ctrl as *const u32)
        }
    }

    pub fn write_ctrl(&self, val: u32) {
        unsafe {
            core::ptr::write_volatile(self.ctrl as *mut u32, val);
        }
    }

    // STATUS
    pub fn read_status(&self) -> u32 {
        unsafe { core::ptr::read_volatile(self.status as *const u32) }
    }

    pub fn write_status(&self, val: u32) {
        unsafe { core::ptr::write_volatile(self.status as *mut u32, val) }
    }

    // FCAL
    pub fn read_fcal(&self) -> u32 {
        unsafe { core::ptr::read_volatile(self.fcal as *const u32) }
    }

    pub fn write_fcal(&self, val: u32) {
        unsafe { core::ptr::write_volatile(self.fcal as *mut u32, val) }
    }

    // FCAH
    pub fn read_fcah(&self) -> u32 {
        unsafe { core::ptr::read_volatile(self.fcah as *const u32) }
    }

    pub fn write_fcah(&self, val: u32) {
        unsafe { core::ptr::write_volatile(self.fcah as *mut u32, val) }
    }

    // FCT
    pub fn read_fct(&self) -> u32 {
        unsafe { core::ptr::read_volatile(self.fct as *const u32) }
    }

    pub fn write_fct(&self, val: u32) {
        unsafe { core::ptr::write_volatile(self.fct as *mut u32, val) }
    }

    // FCTTV
    pub fn read_fcttv(&self) -> u32 {
        unsafe { core::ptr::read_volatile(self.fcttv as *const u32) }
    }

    pub fn write_fcttv(&self, val: u32) {
        unsafe { core::ptr::write_volatile(self.fcttv as *mut u32, val) }
    }

    // EERD
    pub fn read_eerd(&self) -> u32 {
        unsafe { core::ptr::read_volatile(self.eerd as *const u32) }
    }

    pub fn write_eerd(&self, val: u32) {
        unsafe { core::ptr::write_volatile(self.eerd as *mut u32, val) }
    }

    // TDBAL
    pub fn read_tdbal(&self) -> u32 {
        unsafe { core::ptr::read_volatile(self.tdbal as *const u32) }
    }

    pub fn write_tdbal(&self, val: u32) {
        unsafe { core::ptr::write_volatile(self.tdbal as *mut u32, val) }
    }

    // TDBAH
    pub fn read_tdbah(&self) -> u32 {
        unsafe { core::ptr::read_volatile(self.tdbah as *const u32) }
    }

    pub fn write_tdbah(&self, val: u32) {
        unsafe { core::ptr::write_volatile(self.tdbah as *mut u32, val) }
    }

    // TDLEN
    pub fn read_tdlen(&self) -> u32 {
        unsafe { core::ptr::read_volatile(self.tdlen as *const u32) }
    }

    pub fn write_tdlen(&self, val: u32) {
        unsafe { core::ptr::write_volatile(self.tdlen as *mut u32, val) }
    }

    // TDH
    pub fn read_tdh(&self) -> u32 {
        unsafe { core::ptr::read_volatile(self.tdh as *const u32) }
    }

    pub fn write_tdh(&self, val: u32) {
        unsafe { core::ptr::write_volatile(self.tdh as *mut u32, val) }
    }

    // TDT
    pub fn read_tdt(&self) -> u32 {
        unsafe { core::ptr::read_volatile(self.tdt as *const u32) }
    }

    pub fn write_tdt(&self, val: u32) {
        unsafe { core::ptr::write_volatile(self.tdt as *mut u32, val) }
    }

    // TCTL
    pub fn read_tctl(&self) -> u32 {
        unsafe { core::ptr::read_volatile(self.tctl as *const u32) }
    }

    pub fn write_tctl(&self, val: u32) {
        unsafe { core::ptr::write_volatile(self.tctl as *mut u32, val) }
    }

    // TIPG
    pub fn read_tipg(&self) -> u32 {
        unsafe { core::ptr::read_volatile(self.tipg as *const u32) }
    }

    pub fn write_tipg(&self, val: u32) {
        unsafe { core::ptr::write_volatile(self.tipg as *mut u32, val) }
    }

    // RDBAL
    pub fn read_rdbal(&self) -> u32 {
        unsafe { core::ptr::read_volatile(self.rdbal as *const u32) }
    }

    pub fn write_rdbal(&self, val: u32) {
        unsafe { core::ptr::write_volatile(self.rdbal as *mut u32, val) }
    }

    // RDBAH
    pub fn read_rdbah(&self) -> u32 {
        unsafe { core::ptr::read_volatile(self.rdbah as *const u32) }
    }

    pub fn write_rdbah(&self, val: u32) {
        unsafe { core::ptr::write_volatile(self.rdbah as *mut u32, val) }
    }

    // RDLEN
    pub fn read_rdlen(&self) -> u32 {
        unsafe { core::ptr::read_volatile(self.rdlen as *const u32) }
    }

    pub fn write_rdlen(&self, val: u32) {
        unsafe { core::ptr::write_volatile(self.rdlen as *mut u32, val) }
    }

    // RDH
    pub fn read_rdh(&self) -> u32 {
        unsafe { core::ptr::read_volatile(self.rdh as *const u32) }
    }

    pub fn write_rdh(&self, val: u32) {
        unsafe { core::ptr::write_volatile(self.rdh as *mut u32, val) }
    }

    // RDT
    pub fn read_rdt(&self) -> u32 {
        unsafe { core::ptr::read_volatile(self.rdt as *const u32) }
    }

    pub fn write_rdt(&self, val: u32) {
        unsafe { core::ptr::write_volatile(self.rdt as *mut u32, val) }
    }

    // RCTL
    pub fn read_rctl(&self) -> u32 {
        unsafe { core::ptr::read_volatile(self.rctl as *const u32) }
    }

    pub fn write_rctl(&self, val: u32) {
        unsafe { core::ptr::write_volatile(self.rctl as *mut u32, val) }
    }



}