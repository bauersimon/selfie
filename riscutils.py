from struct import unpack, pack
from numpy import uint64, int64
from sys import argv
from enum import Enum, auto

#---------------#
### constants ###
#---------------#

class Format(Enum):
    I_FORMAT = auto()
    S_FORMAT = auto()
    R_FORMAT = auto()
    U_FORMAT = auto()
    B_FORMAT = auto()
    J_FORMAT = auto()

class Opcode(Enum):
    LD     = 3
    IMM    = 19
    SD     = 35
    OP     = 51
    LUI    = 55
    BRANCH = 99
    JALR   = 103
    JAL    = 111
    SYSTEM = 115

class Reg(Enum):
    ZERO  = 0
    RA  = 1
    SP  = 2
    GP  = 3
    TP  = 4
    T0  = 5
    T1  = 6
    T2  = 7
    FP  = 8
    S1  = 9
    A0  = 10
    A1  = 11
    A2  = 12
    A3  = 13
    A4  = 14
    A5  = 15
    A6  = 16
    A7  = 17
    S2  = 18
    S3  = 19
    S4  = 20
    S5  = 21
    S6  = 22
    S7  = 23
    S8  = 24
    S9  = 25
    S10 = 26
    S11 = 27
    T3  = 28
    T4  = 29
    T5  = 30
    T6  = 31

format_for_opcode = {
    Opcode.LD    : Format.I_FORMAT,
    Opcode.IMM   : Format.I_FORMAT,
    Opcode.SD    : Format.S_FORMAT,
    Opcode.OP    : Format.R_FORMAT,
    Opcode.LUI   : Format.U_FORMAT,
    Opcode.BRANCH: Format.B_FORMAT,
    Opcode.JALR  : Format.I_FORMAT,
    Opcode.JAL   : Format.J_FORMAT,
    Opcode.SYSTEM: Format.I_FORMAT
}

class F3code(Enum):
    NOP   = auto()
    ADDI  = auto()
    ADD   = auto()
    SUB   = auto()
    MUL   = auto()
    DIVU  = auto()
    REMU  = auto()
    SLTU  = auto()
    LD    = auto()
    SD    = auto()
    BEQ   = auto()
    JALR  = auto()
    ECALL = auto()

class F7code(Enum):
    ADD  = auto()
    MUL  = auto()
    SUB  = auto()
    DIVU = auto()
    REMU = auto()
    SLTU = auto()

# non-unique constans #

F3_NOP   = 0
F3_ADDI  = 0
F3_ADD   = 0
F3_SUB   = 0
F3_MUL   = 0
F3_DIVU  = 5
F3_REMU  = 7
F3_SLTU  = 3
F3_LD    = 3
F3_SD    = 3
F3_BEQ   = 0
F3_JALR  = 0
F3_ECALL = 0

F7_ADD  = 0
F7_MUL  = 1
F7_SUB  = 32
F7_DIVU = 1
F7_REMU = 1
F7_SLTU = 0

# machine #

WORDSIZE = 8
INSTRUCTIONSIZE = 4
WORDSHORT = 'Q' # according to unpack doc

#-------------#
### helpers ###
#-------------#

def get_word(data, at: int) -> uint64:
    """return one word (word-aligned)"""
    if at % WORDSIZE:
        at = WORDSIZE * int(at / WORDSIZE)
    return uint64(unpack(WORDSHORT, data[at : at + WORDSIZE])[0])

def sign_extend(n: uint64, bits: int) -> uint64:
    if (n < (uint64(2)**uint64(bits-1))):
        return (n)
    else:
        return (n - (uint64(2)**uint64(bits)))

def uint64_to_int(n: uint64) -> int:
    return unpack('q',pack('Q', n))[0]

def get_bits(data: uint64, lsb: int, off: int) -> uint64:
    return (data << uint64(8*WORDSIZE - (lsb + off))) >> uint64(8*WORDSIZE - off)

#--------------------------#
### binary, instructions ###
#--------------------------#

class Binary:

    def __init__(self, filename: str):
        with open(filename, 'rb') as file:
            data = file.read()
        binary_offset = int(get_word(data,  9 * WORDSIZE))
        self.code_length = int(get_word(data, 15 * WORDSIZE))
        self.binary_length = int(get_word(data, 12 * WORDSIZE))
        if binary_offset > len(data):
            raise Exception('extracted funny binary offset, check RISC-header')
        self.binary= data[binary_offset: binary_offset + self.binary_length]
        self.name = filename

    def get_instruction(self, pc: int):
        """return Instruction at program counter"""
        ins = get_word(self.binary, pc)
        if pc % WORDSIZE:
            ins = ins >> uint64(INSTRUCTIONSIZE*8)
        return Instruction(ins, pc)

    def __iter__(self):
        for pc in range(0, self.code_length, INSTRUCTIONSIZE):
            try:
                yield self.get_instruction(pc)
            except ValueError:
                raise ValueError('unknown instruction at pc: {}'.format(pc))

    def __str__(self):
        out = []
        for pc in range(0, self.code_length, INSTRUCTIONSIZE):
            try:
                out.append(str(self.get_instruction(pc)))
            except ValueError:
                raise ValueError('unknown instruction at pc: {}'.format(pc))
        for pc in range(self.code_length, self.binary_length, WORDSIZE):
            out.append('{}: .quad {}'.format(hex(pc), hex(get_word(self.binary, pc))))
        return '\n'.join(out)

class Instruction:

    def __init__(self, ins, pc):
        self.pc = pc
        self.opcode = None
        self.rs1 = None
        self.rs2 = None
        self.rd = None
        self.imm = None
        self.funct3 = None
        self.funct7 = None
        self.format = None

        self.opcode= Opcode(get_bits(ins, 0, 7))
        self.format = format_for_opcode[self.opcode]

        if self.format is Format.I_FORMAT:
            self.funct7 = 0
            self.rs2 = 0
            self.rs1 = Reg(get_bits(ins, 15, 5))
            self.funct3 = get_bits(ins, 12, 3)
            self.rd = Reg(get_bits(ins, 7, 5))
            self.imm = sign_extend(get_bits(ins, 20, 12), 12)

        elif self.format is Format.S_FORMAT:
            self.funct7 = 0
            self.rs2 = Reg(get_bits(ins, 20, 5))
            self.rs1 = Reg(get_bits(ins, 15, 5))
            self.funct3 = get_bits(ins, 12, 3)
            self.rd = 0
            self.imm = sign_extend((get_bits(ins, 25, 7) << uint64(5)) + get_bits(ins, 7, 5), 12)

        elif self.format is Format.R_FORMAT:
            self.funct7 = get_bits(ins, 25, 7)
            self.rs2 = Reg(get_bits(ins, 20, 5))
            self.rs1 = Reg(get_bits(ins, 15, 5))
            self.funct3 = get_bits(ins, 12, 3)
            self.rd = Reg(get_bits(ins, 7, 5))
            self.imm = 0

        elif self.format is Format.B_FORMAT:
            self.funct7 = 0
            self.rs2 = Reg(get_bits(ins, 20, 5))
            self.rs1 = Reg(get_bits(ins, 15, 5))
            self.funct3 = get_bits(ins, 12, 3)
            self.rd = 0
            i1 = get_bits(ins, 31, 1)
            i2 = get_bits(ins, 25, 6)
            i3 = get_bits(ins, 8, 4)
            i4 = get_bits(ins, 7, 1)
            self.imm = sign_extend(((((((i1 << uint64(1)) + i4) << uint64(6)) + i2) << uint64(4)) + i3) << uint64(1), 13) # added trailing zero
            self.instruction_offset = uint64_to_int(uint64(int64(self.imm) / INSTRUCTIONSIZE)) # signed division

        elif self.format is Format.J_FORMAT:
            self.funct7 = 0
            self.rs2 = 0
            self.rs1 = 0
            self.funct3 = 0
            self.rd = Reg(get_bits(ins, 7, 5))
            i1 = get_bits(ins, 31, 1)
            i2 = get_bits(ins, 21, 10)
            i3 = get_bits(ins, 20, 1)
            i4 = get_bits(ins, 12, 8)
            self.imm = sign_extend(((((((i1 << uint64(8)) + i4) << uint64(1)) + i3) << uint64(10)) + i2) << uint64(1), 21) # added trailing zero
            self.instruction_offset = uint64_to_int(uint64(int64(self.imm) / INSTRUCTIONSIZE)) # signed division

        elif self.format is Format.U_FORMAT:
            self.funct7 = 0
            self.rs2 = 0
            self.rs1 = 0
            self.funct3 = 0
            self.rd = Reg(get_bits(ins, 7, 5))
            self.imm_unextended = get_bits(ins, 12, 20)
            self.imm = sign_extend(self.imm_unextended, 20)

        if self.opcode is Opcode.IMM:
            if self.funct3 == F3_ADDI:
                self.funct3 = F3code.ADDI
        elif self.opcode is Opcode.LD:
            if self.funct3 == F3_LD:
                self.funct3 = F3code.LD
        elif self.opcode is Opcode.SD:
            if self.funct3 == F3_SD:
                self.funct3 = F3code.SD
        elif self.opcode is Opcode.OP:
            if self.funct3 == F3_ADD:
                self.funct3 = F3code.ADD
                if self.funct7 == F7_ADD:
                    self.funct7 = F7code.ADD
                elif self.funct7 == F7_SUB:
                    self.funct7 = F7code.SUB
                elif self.funct7 == F7_MUL:
                    self.funct7 = F7code.MUL
            elif self.funct3 == F3_DIVU:
                self.funct3 = F3code.DIVU
                if self.funct7 == F7_DIVU:
                    self.funct7 = F7code.DIVU
            elif self.funct3 == F3_REMU:
                self.funct3 = F3code.REMU
                if self.funct7 == F7_REMU:
                    self.funct7 = F7code.REMU
            elif self.funct3 == F3_SLTU:
                self.funct3 = F3code.SLTU
                if self.funct7 == F7_SLTU:
                    self.funct7 = F7code.SLTU
        elif self.opcode is Opcode.BRANCH:
            if self.funct3 == F3_BEQ:
                self.funct3 = F3code.BEQ
        elif self.opcode is Opcode.JALR:
            if self.funct3 == F3_JALR:
                self.funct3 = F3code.JALR
        elif self.opcode is Opcode.SYSTEM:
            if self.funct3 == F3_ECALL:
                self.funct3 = F3code.ECALL

    def __str__(self):
        imm = uint64_to_int(self.imm)
        if self.format is Format.I_FORMAT:
            if self.funct3 is F3code.ADDI and self.rd is Reg.ZERO and self.rs1 is Reg.ZERO:
                return '{}: nop'.format(hex(self.pc))
            elif self.funct3 is F3code.ECALL:
                return '{}: ecall'.format(hex(self.pc))
            elif self.funct3 is F3code.LD or self.funct3 is F3code.JALR:
                return('{}: {} ${},{}(${})'.format(hex(self.pc), self.funct3.name, self.rd.name,  imm, self.rs1.name))
            return('{}: {} ${},${},{}'.format(hex(self.pc), self.funct3.name, self.rd.name, self.rs1.name, imm))
        elif self.format is Format.S_FORMAT:
            return('{}: {} ${},{}(${})'.format(hex(self.pc), self.funct3.name, self.rs2.name, imm, self.rs1.name))
        elif self.format is Format.R_FORMAT:
            return('{}: {} ${},${},${}'.format(hex(self.pc), self.funct7.name, self.rd.name, self.rs1.name, self.rs2.name))
        elif self.format is Format.U_FORMAT:
            return('{}: {} ${},{}'.format(hex(self.pc), self.opcode.name, self.rd.name, hex(self.imm_unextended)))
        elif self.format is Format.B_FORMAT:
            return('{}: {} ${},${},{}[{}]'.format(hex(self.pc), self.funct3.name, self.rs1.name, self.rs2.name, self.instruction_offset, hex(imm + self.pc)))
        elif self.format is Format.J_FORMAT:
            return('{}: {} ${},{}[{}]'.format(hex(self.pc), self.opcode.name, self.rd.name, self.instruction_offset, hex(imm + self.pc)))
        return '.quad'

if __name__ == '__main__':
    print(Binary(argv[1]))