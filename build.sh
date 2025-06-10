PROJECT_ROOT="$(cd "$(dirname "$0")"; pwd)"   # 当前脚本所在的项目根目录
BUILD_DIR="$PROJECT_ROOT/build"
EXECUTABLE="$BUILD_DIR/my-project"       # 替换为你的实际可执行文件名

rm -rf "$BUILD_DIR"
mkdir "$BUILD_DIR"
cd "$BUILD_DIR"
cmake ..
make

cd "$PROJECT_ROOT"
"$EXECUTABLE"