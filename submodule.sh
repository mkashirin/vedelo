git rm -rf runtime/third-party/nv-codec-headers-n12.0.16.1
rm -rf .git/modules/runtime/third-party/nv-codec-headers-n12.0.16.1

git submodule add https://github.com/FFmpeg/nv-codec-headers runtime/third-party/nv-codec-headers-n12.0.16.1
git -C runtime/third-party/nv-codec-headers-n12.0.16.1 checkout n12.0.16.1
