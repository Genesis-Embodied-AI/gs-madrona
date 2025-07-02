cd build
make -j
cd ..
# .envrc
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6:$LD_PRELOAD"
LD_PRELOAD=/usr/local/lib/librenderdoc.so python ./src/bridge/examples/simple_demo.py
