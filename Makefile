# Compiler flags
CFLAGS := -w -O3 -m64 -D'main(a,b)=main(int argc, char** argv)' -Duint64_t='unsigned long long'

# Compile selfie.c into selfie executable
selfie: selfie.c
	$(CC) $(CFLAGS) $< -o $@

# Consider these targets as targets, not files
.PHONY : test sat all clean vipster

# Test self-compilation, self-execution, and self-hosting
test: selfie
	./selfie -c selfie.c -o selfie1.m -s selfie1.s -m 4 -c selfie.c -o selfie2.m -s selfie2.s
	diff -q selfie1.m selfie2.m
	diff -q selfie1.s selfie2.s
	./selfie -c selfie.c -o selfie.m -m 6 -l selfie.m -m 1
	./selfie -c selfie.c -o selfie3.m -s selfie3.s -m 16 -l selfie3.m -y 8 -l selfie3.m -y 8 -c selfie.c -o selfie4.m -s selfie4.s
	diff -q selfie3.m selfie4.m
	diff -q selfie3.s selfie4.s
	diff -q selfie1.m selfie3.m
	diff -q selfie1.s selfie3.s
	./selfie -c selfie.c -o selfie5.m -s selfie5.s -min 128 -l selfie5.m -y 4 -l selfie5.m -y 4 -c selfie.c -o selfie6.m -s selfie6.s
	diff -q selfie5.m selfie6.m
	diff -q selfie5.s selfie6.s
	diff -q selfie3.m selfie5.m
	diff -q selfie3.s selfie5.s
	./selfie -c -mob 1

# Test SAT solver
sat: selfie
	./selfie -sat manuscript/cnfs/rivest.cnf
	./selfie -c selfie.c -m 1 -sat manuscript/cnfs/rivest.cnf

# Test vipster
vipster: selfie
	./selfie -c manuscript/code/hello-world.c -o h1.m -s h1.s
	./selfie -c selfie.c -v 4 -c manuscript/code/hello-world.c -o h2.m -s h2.s
	diff -q h1.m h2.m
	diff -q h1.s h2.s
	./selfie -c selfie.c -o selfie1.m -s selfie1.s -m 4 -c selfie.c -o selfie2.m -s selfie2.s
	diff -q selfie1.m selfie2.m
	diff -q selfie1.s selfie2.s
	./selfie -c selfie.c -v 4 -c selfie.c -o selfie3.m -s selfie3.s
	diff -q selfie1.m selfie3.m
	diff -q selfie1.s selfie3.s

# Test everything
all: test sat vipster

# Clean up
clean:
	rm -rf *.m
	rm -rf *.s
	rm -rf selfie
	rm -rf selfie.exe
