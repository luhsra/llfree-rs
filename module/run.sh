qemu-system-x86_64 -m 28G -smp 4 \
  -enable-kvm \
  -hda hda.qcow2 \
  -serial mon:stdio \
  -nic user,hostfwd=tcp:127.0.0.1:2222-:22 \
  -nographic \
  # -nic user,id=nic0,smb=$PWD \
  # -kernel 'vmlinuz' \
  # -append 'root=/dev/sda1 console=ttyS0' \
  # -curses \
  # -serial tcp::6666,server,nowait \
  # -kernel ../linux/arch/x86/boot/bzImage \
  # -append 'root=/dev/sda1 ro console=ttyS0 kgdbwait kgdboc=ttyS1 nokaslr' \
  # -append 'console=ttyS0' \
