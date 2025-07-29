class WanParameter{
    public:
        uint64_t rtt = 20; // rtt in ms
        uint64_t comm_bytes_per_ms = 5 * 1024 * 1024 / 1000; // 5 MB/s 
};