#!bash
mkdir cmake-build-emscript
emcmake cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./cmake-build-emscript
emmake cmake --build ./cmake-build-emscript -j 32
scp ./cmake-build-emscript/COSC-4P80-Assignment-3.* administrator@supercomputer:/var/www/tpgc.me/playground/self_organizing_maps/motors_galore/
