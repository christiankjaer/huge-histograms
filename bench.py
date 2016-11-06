import subprocess
size = 10000000
for i in xrange(16, 4097, 16):
  subprocess.check_call(["./histtest", str(size), str(i)])
