#!/usr/bin/env python

# Copyright 2020 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function
import argparse, multiprocessing, os, shutil, subprocess

def driver(thread_count):
    root_dir = os.path.dirname(os.path.realpath(__file__))

    if thread_count is None:
        thread_count = multiprocessing.cpu_count()

    fftw_basename = 'fftw-3.3.8'
    fftw_tar = os.path.join(root_dir, '%s.tar.gz' % fftw_basename)
    fftw_src = os.path.join(root_dir, fftw_basename)
    fftw_build = os.path.join(root_dir, fftw_basename, 'build')
    fftw_install = os.path.join(root_dir, fftw_basename, 'install')

    os.remove(fftw_tar)
    shutil.rmtree(fftw_src)
    subprocess.check_call(['curl', '-O', 'http://www.fftw.org/fftw-3.3.8.tar.gz'], cwd=root_dir)
    subprocess.check_call(['tar', 'xfz', fftw_tar], cwd=root_dir)
    os.mkdir(fftw_build)
    subprocess.check_call([os.path.join(fftw_src, 'configure'), '--enable-openmp', '--disable-mpi', '--with-pic', '--prefix=%s' % fftw_install], cwd=fftw_build)
    subprocess.check_call(['make', 'install', '-j%s' % thread_count], cwd=fftw_build)

    with open(os.path.join(root_dir, 'env.sh'), 'w') as f:
        f.write('export INCLUDE_PATH="$INCLUDE_PATH:%s"' % os.path.join(fftw_install, 'include'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Install Regent FFT library.')
    parser.add_argument(
        '-j', dest='thread_count', nargs='?', type=int,
        help='Number threads used to compile.')
    args = parser.parse_args()
    driver(**vars(args))
