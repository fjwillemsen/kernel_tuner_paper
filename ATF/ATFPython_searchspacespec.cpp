auto block_size_x = atf::tuning_parameter("block_size_x", {1, 2, 4, 8, 16, 32});
auto block_size_y = atf::tuning_parameter("block_size_y", {32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256});
auto block_size_z = atf::tuning_parameter("block_size_z", {1});
auto tile_size_x = atf::tuning_parameter("tile_size_x", {1, 2, 3, 4});
auto tile_size_y = atf::tuning_parameter("tile_size_y", {1, 2, 3, 4, 5, 6, 7, 8});
auto tile_stride_x = atf::tuning_parameter("tile_stride_x", {0, 1});
auto tile_stride_y = atf::tuning_parameter("tile_stride_y", {0, 1});
auto loop_unroll_factor_channel = atf::tuning_parameter("loop_unroll_factor_channel", {0}, [&](auto loop_unroll_factor_channel){ return (32 <= block_size_x * block_size_y <= 1024) && (tile_size_x > 1 || tile_stride_x == 0) && (tile_size_y > 1 || tile_stride_y == 0); });

auto tuner = atf::tuner().silent(true).tuning_parameters(block_size_x, block_size_y, block_size_z, tile_size_x, tile_size_y, tile_stride_x, tile_stride_y, loop_unroll_factor_channel);