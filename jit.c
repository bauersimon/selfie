uint64_t main() {
    uint64_t (*s)();

    s = malloc(8);
    *s = 141179876279571;
    // ADDI $A0 $zero 6
    // JALR $zero $RA 0

    return s();
}