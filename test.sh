LIBTROCH=$(python -c "from vedelo import *; print(PYTORCH)")/lib \
LD_LIBRARY_PATH=$LIBTORCH:$LD_LIBRARY_PATH \
./target/release/vedelo \
    -m finetuned/Vedelo-V1S.torchscript \
    -s dataset/test/video_test.mp4 \
    -d static/video_track.mp4 \
    --conf 0.5 \
    --iou-thresh 0.2 \
    --max-age 90

