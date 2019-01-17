from struct import unpack, pack
from numpy import uint64, int64
from re import fullmatch
from enum import Enum, auto
from os import path
from sys import exit
import argparse

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
    ld     = 3
    imm    = 19
    sd     = 35
    op     = 51
    lui    = 55
    branch = 99
    jalr   = 103
    jal    = 111
    system = 115

class Reg(Enum):
    zero  = 0
    ra  = 1
    sp  = 2
    gp  = 3
    tp  = 4
    t0  = 5
    t1  = 6
    t2  = 7
    fp  = 8
    s1  = 9
    a0  = 10
    a1  = 11
    a2  = 12
    a3  = 13
    a4  = 14
    a5  = 15
    a6  = 16
    a7  = 17
    s2  = 18
    s3  = 19
    s4  = 20
    s5  = 21
    s6  = 22
    s7  = 23
    s8  = 24
    s9  = 25
    s10 = 26
    s11 = 27
    t3  = 28
    t4  = 29
    t5  = 30
    t6  = 31

format_for_opcode = {
    Opcode.ld    : Format.I_FORMAT,
    Opcode.imm   : Format.I_FORMAT,
    Opcode.sd    : Format.S_FORMAT,
    Opcode.op    : Format.R_FORMAT,
    Opcode.lui   : Format.U_FORMAT,
    Opcode.branch: Format.B_FORMAT,
    Opcode.jalr  : Format.I_FORMAT,
    Opcode.jal   : Format.J_FORMAT,
    Opcode.system: Format.I_FORMAT
}

reg_from_string = { # enum maps from register number
    "zero" : Reg.zero,
    "ra"   : Reg.ra,
    "sp"   : Reg.sp,
    "gp"   : Reg.gp,
    "tp"   : Reg.tp,
    "t0"   : Reg.t0,
    "t1"   : Reg.t1,
    "t2"   : Reg.t2,
    "fp"   : Reg.fp,
    "s1"   : Reg.s1,
    "a0"   : Reg.a0,
    "a1"   : Reg.a1,
    "a2"   : Reg.a2,
    "a3"   : Reg.a3,
    "a4"   : Reg.a4,
    "a5"   : Reg.a5,
    "a6"   : Reg.a6,
    "a7"   : Reg.a7,
    "s2"   : Reg.s2,
    "s3"   : Reg.s3,
    "s4"   : Reg.s4,
    "s5"   : Reg.s5,
    "s6"   : Reg.s6,
    "s7"   : Reg.s7,
    "s8"   : Reg.s8,
    "s9"   : Reg.s9,
    "s10"  : Reg.s10,
    "s11"  : Reg.s11,
    "t3"   : Reg.t3,
    "t4"   : Reg.t4,
    "t5"   : Reg.t5,
    "t6"   : Reg.t6,
}

class F3code(Enum):
    nop   = "nop"
    addi  = "addi"
    add   = "add"
    sub   = "sub"
    mul   = "mul"
    divu  = "divu"
    remu  = "remu"
    sltu  = "sltu"
    ld    = "ld"
    sd    = "sd"
    beq   = "beq"
    jalr  = "jalr"
    ecall = "ecall"

class F7code(Enum):
    add  = "add"
    mul  = "mul"
    sub  = "sub"
    divu = "divu"
    remu = "remu"
    sltu = "sltu"

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

#-------------#
### helpers ###
#-------------#

def get_word(data, at: int) -> uint64:
    """return one word (word-aligned)"""
    if at % WORDSIZE:
        at = WORDSIZE * int(at / WORDSIZE)
    return uint64(unpack('Q', data[at : at + WORDSIZE])[0]) # unsigned quadword (struct)

def sign_extend(n: uint64, bits: int) -> uint64:
    if (n < (uint64(2)**uint64(bits-1))):
        return (n)
    else:
        return (n - (uint64(2)**uint64(bits)))

def sign_shrink(n: uint64, bits: int) -> uint64:
    return get_bits(n, 0, bits)

def uint64_to_int(n: uint64) -> int:
    return unpack('q',pack('Q', n))[0] # (un)signed quadword (struct)

def int_to_uint64(n: int) -> uint64:
    return unpack('Q',pack('q', n))[0] # (un)signed quadword (struct)

def get_bits(data: uint64, lsb: int, off: int) -> uint64:
    return (data << uint64(8*WORDSIZE - (lsb + off))) >> uint64(8*WORDSIZE - off)

def write_out(text: str, filename: str):
    with open(filename, 'w') as file:
        file.write(text)

def to_hex(n: int) -> str:
    return '0x' + format(n, 'X')

#--------------------------#
### binary, instructions ###
#--------------------------#

class Binary:

    def __init__(self):
        self.instructions = []
        self.data_segment = []

    @classmethod
    def from_binary(cls, filename: str):
        item = cls()
        with open(filename, 'rb') as file:
            data = file.read()

        binary_offset = int(get_word(data,  9 * WORDSIZE))
        item.code_length = int(get_word(data, 15 * WORDSIZE))
        item.binary_length = int(get_word(data, 12 * WORDSIZE))

        if binary_offset > len(data) or item.code_length > len(data) or item.binary_length > len(data):
            # roughly validate
            raise Exception('invalid RISC-header')
        binary= data[binary_offset: binary_offset + item.binary_length]

        for pc in range(0, item.code_length, INSTRUCTIONSIZE):
            ins = get_word(binary, pc)
            if pc % WORDSIZE:
                ins = ins >> uint64(INSTRUCTIONSIZE*8)
            try:
                item.instructions.append(Instruction.from_binary(ins, pc))
            except ValueError:
                raise ValueError('unknown instruction at pc: {}'.format(pc))
        
        for pc in range(item.code_length, item.binary_length, WORDSIZE):
            item.data_segment.append(get_word(binary, pc))

        return item

    @classmethod
    def from_assembly(cls, filename: str):
        item = cls()
        with open(filename, 'r') as file:
            data = file.read()
        lines = data.split('\n')

        for line in lines:
            if not line == '':
                if not '.quad' in line:
                    item.instructions.append(Instruction.from_assembly(line))
                else:
                    item.data_segment.append(int(fullmatch(r'^.*(?P<data>0x[A-F0-9]+).*$',line).groupdict()['data'], 0)) # extract hex value first
        
        item.code_length = len(item.instructions) * INSTRUCTIONSIZE
        item.binary_length = item.code_length + len(item.data_segment) * WORDSIZE

        return item

    def get_instruction(self, pc: int):
        if pc < self.code_length:
            return self.instructions[int(pc / INSTRUCTIONSIZE)]
        elif pc < self.binary_length:
            return self.data_segment[int((pc-self.code_length) / WORDSIZE)]

    def __str__(self):
        out = []
        for pc in range(0, self.code_length, INSTRUCTIONSIZE):
            out.append(str(self.get_instruction(pc)))

        for pc in range(self.code_length, self.binary_length, WORDSIZE):
            out.append('{}: .quad {}'.format(to_hex(pc), to_hex(self.get_instruction(pc))))

        return '\n'.join(out)

class Instruction:

    def __init__(self, pc):
        self.pc = pc
        self.opcode = None
        self.rs1 = None
        self.rs2 = None
        self.rd = None
        self.imm = None
        self.funct3 = None
        self.funct7 = None
        self.format = None

    @classmethod
    def from_assembly(cls, assembly: str):
        match = fullmatch(r'^(?P<pc>(?:0x|0X)[a-fA-F0-9]+):\s*(?P<type>[A-Za-z]*)\s*(?P<detail>.*?)\s*$', assembly) # match pc and instruction
        if not match:
            raise ValueError("cannot parse assembly: \"{}\"".format(assembly))
        i = cls(int(match.groupdict()["pc"], 0)) # init empty instruction from pc
        detail = match.groupdict()["detail"]

        try: # to get f3code from type
            i.funct3 = F3code(match.groupdict()["type"])
        except ValueError: # lui or jal not present in f3code
            i.funct3 = None
            if match.groupdict()["type"] == "lui":
                i.format = Format.U_FORMAT
                i.opcode = Opcode.lui
            elif match.groupdict()["type"] == "jal":
                i.format = Format.J_FORMAT
                i.opcode = Opcode.jal

        if i.funct3: # infer in additional information from f3code
            if i.funct3 is F3code.ld or i.funct3 is F3code.addi or i.funct3 is F3code.nop:
                i.format = Format.I_FORMAT
                if i.funct3 is F3code.ld:
                    i.opcode = Opcode.ld
                elif i.funct3 is F3code.addi or i.funct3 is F3code.nop:
                    i.opcode = Opcode.imm
            elif i.funct3 is F3code.sd:
                i.format = Format.S_FORMAT
                i.opcode = Opcode.sd
            elif i.funct3 is F3code.add or i.funct3 is F3code.sub or i.funct3 is F3code.mul or i.funct3 is F3code.remu or i.funct3 is F3code.sltu or i.funct3 is F3code.divu:
                i.format = Format.R_FORMAT
                i.opcode = Opcode.op
                i.funct7 = F7code(i.funct3.name.lower()) # get enum from f3 string
            elif i.funct3 is F3code.beq:
                i.format = Format.B_FORMAT
                i.opcode = Opcode.branch
            elif i.funct3 is F3code.jalr:
                i.format = Format.I_FORMAT
                i.opcode = Opcode.jalr
            elif i.funct3 is F3code.ecall:
                i.format = Format.I_FORMAT
                i.opcode = Opcode.system
                return i # no registers for syscall

        if i.format is Format.I_FORMAT:
            if i.opcode is Opcode.ld or i.opcode is Opcode.jalr: # special format for ld and jalr
                match = fullmatch(r'^\$(?P<rd>[a-zA-Z0-9]+),\s*(?P<imm>[\-0-9]+)\s*\(\$(?P<rs1>[a-zA-Z0-9]+)\)$', detail)
            else:
                match = fullmatch(r'^\$(?P<rd>[a-zA-Z0-9]+),\s*\$(?P<rs1>[a-zA-Z0-9]+),\s*(?P<imm>[\-0-9]+)$', detail)
            if not match:
                raise ValueError("cannot parse assembly: \"{}\"".format(assembly))
            i.rd = reg_from_string[match.groupdict()["rd"]]
            i.rs1 = reg_from_string[match.groupdict()["rs1"]]
            i.imm = int(match.groupdict()["imm"],10)
        elif i.format is Format.S_FORMAT:
            match = fullmatch(r'^\$(?P<rs2>[a-zA-Z0-9]+),\s*(?P<imm>[\-0-9]+)\s*\(\$(?P<rs1>[a-zA-Z0-9]+)\)$', detail)
            if not match:
                raise ValueError("cannot parse assembly: \"{}\"".format(assembly))
            i.rs1 = reg_from_string[match.groupdict()["rs1"]]
            i.rs2 = reg_from_string[match.groupdict()["rs2"]]
            i.imm = int(match.groupdict()["imm"],10)
        elif i.format is Format.R_FORMAT:
            match = fullmatch(r'^\$(?P<rd>[a-zA-Z0-9]+),\s*\$(?P<rs1>[a-zA-Z0-9]+),\s*\$(?P<rs2>[a-zA-Z0-9]+)$', detail)
            if not match:
                raise ValueError("cannot parse assembly: \"{}\"".format(assembly))
            i.rd = reg_from_string[match.groupdict()["rd"]]
            i.rs1 = reg_from_string[match.groupdict()["rs1"]]
            i.rs2 = reg_from_string[match.groupdict()["rs2"]]
        elif i.format is Format.U_FORMAT:
            match = fullmatch(r'^\$(?P<rd>[a-zA-Z0-9]+),\s*(?P<imm>0x[A-F0-9]+)$', detail)
            if not match:
                raise ValueError("cannot parse assembly: \"{}\"".format(assembly))
            i.rd = reg_from_string[match.groupdict()["rd"]]
            i.imm = int(match.groupdict()["imm"],0)
        elif i.format is Format.B_FORMAT:
            match = fullmatch(r'^\$(?P<rs1>[a-zA-Z0-9]+),\s*\$(?P<rs2>[a-zA-Z0-9]+),\s*(?P<rel>[\-0-9]+)\s*\[(?P<abs>0x[A-F0-9]+)\]$', detail)
            if not match:
                raise ValueError("cannot parse assembly: \"{}\"".format(assembly))
            i.rs1 = reg_from_string[match.groupdict()["rs1"]]
            i.rs2 = reg_from_string[match.groupdict()["rs2"]]
            i.imm = int(match.groupdict()["abs"],0) - i.pc
            i.instruction_offset = int(match.groupdict()["rel"],10)
        elif i.format is Format.J_FORMAT:
            match = fullmatch(r'^\$(?P<rd>[a-zA-Z0-9]+),\s*(?P<rel>[\-0-9]+)\s*\[(?P<abs>0x[A-F0-9]+)\]$', detail)
            if not match:
                raise ValueError("cannot parse assembly: \"{}\"".format(assembly))
            i.rd = reg_from_string[match.groupdict()["rd"]]
            i.imm = int(match.groupdict()["abs"],0) - i.pc
            i.instruction_offset = int(match.groupdict()["rel"],10)

        return i

    @classmethod
    def from_binary(cls, ins: uint64, pc: int):
        i = cls(pc) # init empty instruction from pc

        i.opcode= Opcode(get_bits(ins, 0, 7))
        i.format = format_for_opcode[i.opcode]

        if i.format is Format.I_FORMAT:
            i.funct7 = 0
            i.rs2 = 0
            i.rs1 = Reg(get_bits(ins, 15, 5))
            i.funct3 = get_bits(ins, 12, 3)
            i.rd = Reg(get_bits(ins, 7, 5))
            i.imm = sign_extend(get_bits(ins, 20, 12), 12)

        elif i.format is Format.S_FORMAT:
            i.funct7 = 0
            i.rs2 = Reg(get_bits(ins, 20, 5))
            i.rs1 = Reg(get_bits(ins, 15, 5))
            i.funct3 = get_bits(ins, 12, 3)
            i.rd = 0
            i.imm = sign_extend((get_bits(ins, 25, 7) << uint64(5)) + get_bits(ins, 7, 5), 12)

        elif i.format is Format.R_FORMAT:
            i.funct7 = get_bits(ins, 25, 7)
            i.rs2 = Reg(get_bits(ins, 20, 5))
            i.rs1 = Reg(get_bits(ins, 15, 5))
            i.funct3 = get_bits(ins, 12, 3)
            i.rd = Reg(get_bits(ins, 7, 5))
            i.imm = 0

        elif i.format is Format.B_FORMAT:
            i.funct7 = 0
            i.rs2 = Reg(get_bits(ins, 20, 5))
            i.rs1 = Reg(get_bits(ins, 15, 5))
            i.funct3 = get_bits(ins, 12, 3)
            i.rd = 0
            i1 = get_bits(ins, 31, 1)
            i2 = get_bits(ins, 25, 6)
            i3 = get_bits(ins, 8, 4)
            i4 = get_bits(ins, 7, 1)
            i.imm = sign_extend(((((((i1 << uint64(1)) + i4) << uint64(6)) + i2) << uint64(4)) + i3) << uint64(1), 13) # added trailing zero
            i.instruction_offset = uint64_to_int(uint64(int64(i.imm) / INSTRUCTIONSIZE)) # signed division

        elif i.format is Format.J_FORMAT:
            i.funct7 = 0
            i.rs2 = 0
            i.rs1 = 0
            i.funct3 = 0
            i.rd = Reg(get_bits(ins, 7, 5))
            i1 = get_bits(ins, 31, 1)
            i2 = get_bits(ins, 21, 10)
            i3 = get_bits(ins, 20, 1)
            i4 = get_bits(ins, 12, 8)
            i.imm = sign_extend(((((((i1 << uint64(8)) + i4) << uint64(1)) + i3) << uint64(10)) + i2) << uint64(1), 21) # added trailing zero
            i.instruction_offset = uint64_to_int(uint64(int64(i.imm) / INSTRUCTIONSIZE)) # signed division

        elif i.format is Format.U_FORMAT:
            i.funct7 = 0
            i.rs2 = 0
            i.rs1 = 0
            i.funct3 = 0
            i.rd = Reg(get_bits(ins, 7, 5))
            i.imm = sign_extend(get_bits(ins, 12, 20), 20)

        if i.opcode is Opcode.imm:
            if i.funct3 == F3_ADDI:
                i.funct3 = F3code.addi
        elif i.opcode is Opcode.ld:
            if i.funct3 == F3_LD:
                i.funct3 = F3code.ld
        elif i.opcode is Opcode.sd:
            if i.funct3 == F3_SD:
                i.funct3 = F3code.sd
        elif i.opcode is Opcode.op:
            if i.funct3 == F3_ADD:
                i.funct3 = F3code.add
                if i.funct7 == F7_ADD:
                    i.funct7 = F7code.add
                elif i.funct7 == F7_SUB:
                    i.funct7 = F7code.sub
                elif i.funct7 == F7_MUL:
                    i.funct7 = F7code.mul
            elif i.funct3 == F3_DIVU:
                i.funct3 = F3code.divu
                if i.funct7 == F7_DIVU:
                    i.funct7 = F7code.divu
            elif i.funct3 == F3_REMU:
                i.funct3 = F3code.remu
                if i.funct7 == F7_REMU:
                    i.funct7 = F7code.remu
            elif i.funct3 == F3_SLTU:
                i.funct3 = F3code.sltu
                if i.funct7 == F7_SLTU:
                    i.funct7 = F7code.sltu
        elif i.opcode is Opcode.branch:
            if i.funct3 == F3_BEQ:
                i.funct3 = F3code.beq
        elif i.opcode is Opcode.jalr:
            if i.funct3 == F3_JALR:
                i.funct3 = F3code.jalr
        elif i.opcode is Opcode.system:
            if i.funct3 == F3_ECALL:
                i.funct3 = F3code.ecall

        if i.imm:
            i.imm = uint64_to_int(i.imm)

        return i

    def __str__(self):
        if self.format is Format.I_FORMAT:
            if self.funct3 is F3code.addi and self.rd is Reg.zero and self.rs1 is Reg.zero:
                return '{}: nop'.format(to_hex(self.pc))
            elif self.funct3 is F3code.ecall:
                return '{}: ecall'.format(to_hex(self.pc))
            elif self.funct3 is F3code.ld or self.funct3 is F3code.jalr:
                return('{}: {} ${},{}(${})'.format(to_hex(self.pc), self.funct3.name, self.rd.name,  self.imm, self.rs1.name))
            return('{}: {} ${},${},{}'.format(to_hex(self.pc), self.funct3.name, self.rd.name, self.rs1.name, self.imm))
        
        elif self.format is Format.S_FORMAT:
            return('{}: {} ${},{}(${})'.format(to_hex(self.pc), self.funct3.name, self.rs2.name, self.imm, self.rs1.name))
        
        elif self.format is Format.R_FORMAT:
            return('{}: {} ${},${},${}'.format(to_hex(self.pc), self.funct7.name, self.rd.name, self.rs1.name, self.rs2.name))
        
        elif self.format is Format.U_FORMAT:
            return('{}: {} ${},{}'.format(to_hex(self.pc), self.opcode.name, self.rd.name, to_hex(sign_shrink(uint64(self.imm),20))))
        
        elif self.format is Format.B_FORMAT:
            return('{}: {} ${},${},{}[{}]'.format(to_hex(self.pc), self.funct3.name, self.rs1.name, self.rs2.name, self.instruction_offset, to_hex(self.imm + self.pc)))
        
        elif self.format is Format.J_FORMAT:
            return('{}: {} ${},{}[{}]'.format(to_hex(self.pc), self.opcode.name, self.rd.name, self.instruction_offset, to_hex(self.imm + self.pc)))
        
        return str(self.pc) # at least return the pc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'convert risc-u binary/assembly', epilog= 'input/ouput types are infered from file extentions')
    parser.add_argument('source', help= 'input file', type= str)
    parser.add_argument('dest', help= 'desired ouput file', type= str)
    args = parser.parse_args() # taken from sys.argv

    if args:
        intype = path.splitext(args.source)[1]
        outype = path.splitext(args.dest)[1]

        if intype == '.s':
            b = Binary.from_assembly(args.source)
        elif intype == '.m':
            b = Binary.from_binary(args.source)
        else:
            print('Unsupported input type: \'{}\''.format(intype))
            exit(-1)

        if outype == '.s':
            write_out(str(b) + '\n', args.dest)
        else:
            print('Unsupported output type: \'{}\''.format(intype))
            exit(-1)