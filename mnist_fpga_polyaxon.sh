#!/bin/bash
source /mnt/inspur_f10a_dev_stack/init_env.sh
python3.6 /mnt/tfdir/tfTestCase/mnist_py/mnist_fpga_polyaxon_dis.py --device=fpga:0
