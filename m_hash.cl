
// Jenkin's 32bit integer hashing function
inline unsigned int hash1(unsigned int val) {
  val = (val+0x7ed55d16) + (val<<12);
  val = (val^0xc761c23c) ^ (val>>19);
  val = (val+0x165667b1) + (val<<5);
  val = (val+0xd3a2646c) ^ (val<<9);
  val = (val+0xfd7046c5) + (val<<3);
  val = (val^0xb55a4f09) ^ (val>>16);
  return val;
}

// Jenkin's 32bit integer hashing function with a different set of magic numbers
inline unsigned int hash2(unsigned int val) {
  val = (val+0x7fb9b1ee) + (val<<12);
  val = (val^0xab35dd63) ^ (val>>19);
  val = (val+0x41ed960d) + (val<<5);
  val = (val+0xc7d0125e) ^ (val<<9);
  val = (val+0x071f9f8f) + (val<<3);
  val = (val^0x55ab55b9) ^ (val>>16);
  return val;
}

// Wang's integer hashing function
inline unsigned int hash3(unsigned int val) {
  val = (val ^ 61) ^ (val >> 16);
  val = val + (val << 3);
  val = val ^ (val >> 4);
  val = val * 0x27d4eb2d;
  val = val ^ (val >> 15);
  return val;
}

// Definitions of how we use our keys
#define ZERO_KEY 0xFFFFFFE
#define EMPTY_KEY 0
#define hash_constant 0x348c3def


inline unsigned int buildHashTableOptimisticIntern(
    int value,
    __global int* hash_table,
    const unsigned int hash_table_size,
    const unsigned int hash_table_offset)
{
  value = (value == 0) ? ZERO_KEY : value;
  unsigned int bucket = hash1(value) % hash_table_size;
  hash_table[hash_table_offset + bucket] = value;

  return bucket;
}

inline void validateHashTableIntern(
    int value,
    __global int* hash_table,
    const unsigned int hash_table_size,
    const unsigned int hash_table_offset,
    __global unsigned char* error)
{
  value = (value == 0) ? ZERO_KEY : value;
  unsigned int bucket = hash1(value) % hash_table_size;
  if (hash_table[hash_table_offset + bucket] != value) {
    *error = 1;
  }
}

inline unsigned int buildHashTablePessimisticIntern(
    int value,
    __global int* hash_table,
    const unsigned int hash_table_size,
    const unsigned int hash_table_offset,
    __global unsigned char* error)
{
  value = (value == 0) ? ZERO_KEY : value;
  // Compute buckets for the cuckoo chain
  unsigned int bucket = hash1(value) % hash_table_size;
  // If the write of the optimistic approach went through, we are done.
  if (hash_table[hash_table_offset + bucket] != value) {
    // Try re-hashing to find a position:
    bucket = hash2(value) % (hash_table_size);
    int test = atomic_cmpxchg(&(hash_table[hash_table_offset + bucket]),
        EMPTY_KEY, value);

    if (test == value || test == EMPTY_KEY)
      return bucket;

    bucket = hash3(value) % (hash_table_size);
    test = atomic_cmpxchg(&(hash_table[hash_table_offset + bucket]),
        EMPTY_KEY, value);

    if (test == value || test == EMPTY_KEY)
      return bucket;

    bucket = hash1(value ^ hash_constant) % (hash_table_size);
    test = atomic_cmpxchg(&(hash_table[hash_table_offset + bucket]),
        EMPTY_KEY, value);

    if (test == value || test == EMPTY_KEY)
      return bucket;

    bucket = hash2(value ^ hash_constant) % (hash_table_size);
    test = atomic_cmpxchg(&(hash_table[hash_table_offset + bucket]),
        EMPTY_KEY, value);

    if (test == value || test == EMPTY_KEY)
      return bucket;

    bucket = hash3(value ^ hash_constant) % (hash_table_size);
    test = atomic_cmpxchg(&(hash_table[hash_table_offset + bucket]),
        EMPTY_KEY, value);

    if (test == value || test == EMPTY_KEY)
      return bucket;

    // Try linear probing to find a position
    for (unsigned int i=0; i<10; ++i) {
      bucket = (bucket + 1) % hash_table_size;
      test = atomic_cmpxchg(&(hash_table[hash_table_offset + bucket]),
          EMPTY_KEY, value);

      if (test == value || test == EMPTY_KEY)
        return bucket;
    }
    // We failed to insert this value :(
    *error = 1;
    return hash_table_size;
  }

  return bucket;
}

inline unsigned int test(int value)
{

	return value;
}



__kernel void buildHashTableOptimistic(
	__global const int* const data,
	__global int* hash_table,
	const  int hash_table_size
) {
	int value = data[get_global_id(0)];
	buildHashTableOptimisticIntern(value, hash_table, hash_table_size, 0);
}

__kernel void validateHashTable(
	__global const int* const data,
	__global int* hash_table,
	const unsigned int hash_table_size,
	__global unsigned char* error
) {
	int value = data[get_global_id(0)];
	validateHashTableIntern(value, hash_table, hash_table_size, 0, error);
}

__kernel void buildHashTablePessimistic(
	__global const int* const data,
	__global int* hash_table,
	const unsigned int hash_table_size,
	__global unsigned char* error
) {
	int value = data[get_global_id(0)];
	buildHashTablePessimisticIntern(value, hash_table, hash_table_size, 0, error);
}