# Environment snapshot — BEFORE oneAPI 2026.1.0-235 → 2026.1.1-325 update

Taken 2026-08-02 by the chat session, pre-update baseline.
OS fact correction for the record: rig is Fedora 44, KDE (kernel 7.1.5-200.fc44.x86_64) — the 2026-08-02 handoff's 'Ubuntu 26.04' was wrong.

## Installed Intel / oneAPI packages
```
intel-audio-firmware-20260622-1.fc44.noarch
intel-compute-runtime-26.22.38646.6-3.fc44.x86_64
intel-gmmlib-22.10.0-1.fc44.x86_64
intel-gpu-firmware-20260622-1.fc44.noarch
intel-igc-2.36.3-3.fc44.x86_64
intel-igc-libs-2.36.3-3.fc44.x86_64
intel-level-zero-26.22.38646.6-3.fc44.x86_64
intel-lpmd-0.1.0-2.fc44.x86_64
intel-media-driver-26.1.5-1.fc44.x86_64
intel-mediasdk-23.2.2-11.fc44.x86_64
intel-ocloc-26.22.38646.6-3.fc44.x86_64
intel-oneapi-common-licensing-2025.3-2025.3.1-15.noarch
intel-oneapi-common-licensing-2026.0-2026.0.0-235.noarch
intel-oneapi-common-oneapi-vars-2025.3-2025.3.1-15.noarch
intel-oneapi-common-oneapi-vars-2026.0-2026.0.0-235.noarch
intel-oneapi-common-vars-2026.0.0-235.noarch
intel-oneapi-compiler-cpp-eclipse-cfg-2026.1-2026.1.0-235.noarch
intel-oneapi-compiler-dpcpp-cpp-2026.1-2026.1.0-235.x86_64
intel-oneapi-compiler-dpcpp-cpp-common-2026.1-2026.1.0-235.noarch
intel-oneapi-compiler-dpcpp-cpp-runtime-2026.1-2026.1.0-235.x86_64
intel-oneapi-compiler-dpcpp-eclipse-cfg-2026.1-2026.1.0-235.noarch
intel-oneapi-compiler-shared-2026.1-2026.1.0-235.x86_64
intel-oneapi-compiler-shared-common-2026.1-2026.1.0-235.noarch
intel-oneapi-compiler-shared-runtime-2026.1-2026.1.0-235.x86_64
intel-oneapi-dev-utilities-2026.0-2026.0.1-16.x86_64
intel-oneapi-dev-utilities-eclipse-cfg-2026.0-2026.0.1-16.noarch
intel-oneapi-dnnl-2026.0-2026.0.1-55.x86_64
intel-oneapi-dnnl-devel-2026.0-2026.0.1-55.x86_64
intel-oneapi-dpcpp-cpp-2026.1-2026.1.0-235.x86_64
intel-oneapi-dpcpp-debugger-2026.1-2026.1.0-70.x86_64
intel-oneapi-icc-eclipse-plugin-cpp-2026.1-2026.1.0-235.noarch
intel-oneapi-libdpstd-devel-2022.13-2022.13.0-107.x86_64
intel-oneapi-mkl-classic-devel-2026.1-2026.1.0-236.x86_64
intel-oneapi-mkl-classic-include-2026.1-2026.1.0-236.x86_64
intel-oneapi-mkl-cluster-2026.1-2026.1.0-236.x86_64
intel-oneapi-mkl-cluster-devel-2026.1-2026.1.0-236.x86_64
intel-oneapi-mkl-core-2026.1-2026.1.0-236.x86_64
intel-oneapi-mkl-core-devel-2026.1-2026.1.0-236.x86_64
intel-oneapi-mkl-devel-2026.1-2026.1.0-236.x86_64
intel-oneapi-mkl-sycl-2026.1-2026.1.0-236.x86_64
intel-oneapi-mkl-sycl-blas-2026.1-2026.1.0-236.x86_64
intel-oneapi-mkl-sycl-data-fitting-2026.1-2026.1.0-236.x86_64
intel-oneapi-mkl-sycl-devel-2026.1-2026.1.0-236.x86_64
intel-oneapi-mkl-sycl-dft-2026.1-2026.1.0-236.x86_64
intel-oneapi-mkl-sycl-include-2026.1-2026.1.0-236.x86_64
intel-oneapi-mkl-sycl-lapack-2026.1-2026.1.0-236.x86_64
intel-oneapi-mkl-sycl-rng-2026.1-2026.1.0-236.x86_64
intel-oneapi-mkl-sycl-sparse-2026.1-2026.1.0-236.x86_64
intel-oneapi-mkl-sycl-stats-2026.1-2026.1.0-236.x86_64
intel-oneapi-mkl-sycl-vm-2026.1-2026.1.0-236.x86_64
intel-oneapi-openmp-2026.1-2026.1.0-235.x86_64
intel-oneapi-openmp-common-2026.1-2026.1.0-235.noarch
intel-oneapi-runtime-compilers-2026.1.0-235.x86_64
intel-oneapi-runtime-compilers-common-2026.1.0-235.noarch
intel-oneapi-runtime-dnnl-2026.0.1-55.x86_64
intel-oneapi-runtime-dpcpp-cpp-2026.1.0-235.x86_64
intel-oneapi-runtime-dpcpp-cpp-common-2026.1.0-235.noarch
intel-oneapi-runtime-dpcpp-sycl-core-2026.1.0-235.x86_64
intel-oneapi-runtime-dpcpp-sycl-opencl-cpu-2026.1.0-235.x86_64
intel-oneapi-runtime-mkl-2026.1.0-236.x86_64
intel-oneapi-runtime-opencl-2026.1.0-235.x86_64
intel-oneapi-runtime-openmp-2026.1.0-235.x86_64
intel-oneapi-runtime-tbb-2023.1.0-151.x86_64
intel-oneapi-runtime-tcm-1.5.0-489.x86_64
intel-oneapi-tbb-2023.1-2023.1.0-151.x86_64
intel-oneapi-tbb-devel-2023.1-2023.1.0-151.x86_64
intel-oneapi-tcm-1.4-1.4.1-445.x86_64
intel-oneapi-tcm-1.5-1.5.0-489.x86_64
intel-oneapi-umf-1.0-1.0.3-17.x86_64
intel-oneapi-umf-1.1-1.1.0-340.x86_64
intel-opencl-26.22.38646.6-3.fc44.x86_64
intel-opencl-clang-15.0.9-2.fc44.x86_64
intel-vpl-gpu-rt-26.1.6-1.fc44.x86_64
intel-vsc-firmware-20260622-1.fc44.noarch
libva-intel-media-driver-26.1.5-1.fc44.x86_64
oneapi-level-zero-1.28.6-1.fc44.x86_64
oneapi-level-zero-zello_world-1.28.6-1.fc44.x86_64
```

## Production binary runtime linkage (build-sycl llama-server, pin 85b7e6556)
```
	libllama-server-impl.so => /home/aaron/workspace/moe-serving/llama.cpp/build-sycl/bin/libllama-server-impl.so (0x00007f3f40a00000)
	libllama-common.so.0 => /home/aaron/workspace/moe-serving/llama.cpp/build-sycl/bin/libllama-common.so.0 (0x00007f3f40400000)
	libmtmd.so.0 => /home/aaron/workspace/moe-serving/llama.cpp/build-sycl/bin/libmtmd.so.0 (0x00007f3f40892000)
	libllama.so.0 => /home/aaron/workspace/moe-serving/llama.cpp/build-sycl/bin/libllama.so.0 (0x00007f3f40000000)
	libggml.so.0 => /home/aaron/workspace/moe-serving/llama.cpp/build-sycl/bin/libggml.so.0 (0x00007f3f410c9000)
	libggml-sycl.so.0 => /home/aaron/workspace/moe-serving/llama.cpp/build-sycl/bin/libggml-sycl.so.0 (0x00007f3f3c200000)
	libggml-cpu.so.0 => /home/aaron/workspace/moe-serving/llama.cpp/build-sycl/bin/libggml-cpu.so.0 (0x00007f3f3be00000)
	libggml-base.so.0 => /home/aaron/workspace/moe-serving/llama.cpp/build-sycl/bin/libggml-base.so.0 (0x00007f3f3c114000)
	libsvml.so => /opt/intel//oneapi/redist/lib/libsvml.so (0x00007f3f39800000)
	libimf.so => /opt/intel//oneapi/redist/lib/libimf.so (0x00007f3f39200000)
	libdnnl.so.3 => /opt/intel//oneapi/redist/lib/libdnnl.so.3 (0x00007f3f33600000)
	libsycl.so.9 => /opt/intel//oneapi/redist/lib/libsycl.so.9 (0x00007f3f33200000)
	libmkl_sycl_blas.so.6 => /opt/intel/oneapi/mkl/2026.1/lib/libmkl_sycl_blas.so.6 (0x00007f3f2f000000)
	libmkl_intel_ilp64.so.3 => /opt/intel/oneapi/mkl/2026.1/lib/libmkl_intel_ilp64.so.3 (0x00007f3f2e000000)
	libmkl_tbb_thread.so.3 => /opt/intel/oneapi/mkl/2026.1/lib/libmkl_tbb_thread.so.3 (0x00007f3f2c000000)
	libmkl_core.so.3 => /opt/intel/oneapi/mkl/2026.1/lib/libmkl_core.so.3 (0x00007f3f27800000)
```

## Pending per cached dnf view at snapshot time (Intel repo metadata may lag Discover)
```
intel-compute-runtime.x86_64          26.22.38646.6-4.fc44       updates
intel-level-zero.x86_64               26.22.38646.6-4.fc44       updates
intel-ocloc.x86_64                    26.22.38646.6-4.fc44       updates
intel-opencl.x86_64                   26.22.38646.6-4.fc44       updates
kernel.x86_64                         7.1.5-201.fc44             updates
kernel-core.x86_64                    7.1.5-201.fc44             updates
kernel-devel.x86_64                   7.1.5-201.fc44             updates
kernel-modules.x86_64                 7.1.5-201.fc44             updates
kernel-modules-core.x86_64            7.1.5-201.fc44             updates
kernel-tools.x86_64                   7.1.5-201.fc44             updates
kernel-tools-libs.x86_64              7.1.5-201.fc44             updates
total pending: 89
```
