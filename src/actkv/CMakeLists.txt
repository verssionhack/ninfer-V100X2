# Active KV tiering ("actkv"). A self-contained, device-agnostic host core for
# keeping native KV history in RAM (and optionally SSD) with LRFU / mean-key
# retrieval, exposed through the abstract MemoryBackend/MemoryTransfer contract.
# Ported from the KVMem common library; it has no dependency on the NInfer
# target/engine and compiles standalone.
add_library(actkv STATIC
  src/host/kvmem_store.cpp
  src/host/kvmem_runtime.cpp
  src/host/raw_kv_store.cpp
  src/host/mean_key_retrieval.cpp
  src/host/graph_recall.cpp
  src/host/memory_contract.cpp
  src/host/kv_content.cpp
  src/host/native_kv_payload.cpp
  src/host/host_kv_storage.cpp
  src/host/spill_io.cpp
  src/host/session_memory.cpp
  src/host/prefix_checkpoint.cpp
  src/host/checkpoint_policy.cpp
  src/host/rope.cpp)
target_include_directories(actkv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(actkv PUBLIC cxx_std_17)
set_target_properties(actkv PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Optional accelerated GPU page-softmax scorer. Folded into actkv so consumers
# that only link actkv get both the host reference and the device path.
if(CMAKE_CUDA_COMPILER)
  add_library(actkv-cuda OBJECT src/cuda/page_softmax.cu)
  target_include_directories(actkv-cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set_target_properties(actkv-cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
  target_sources(actkv PRIVATE $<TARGET_OBJECTS:actkv-cuda>)
  target_link_libraries(actkv PUBLIC CUDA::cudart)
endif()
