import multiprocessing

def chunk_stats(sub_arr):
    return min(sub_arr), max(sub_arr), sum(sub_arr)

def parallel_reduction(arr):
    n = min(multiprocessing.cpu_count(), len(arr))
    size = len(arr) // n

    chunks = [arr[i*size:(i+1)*size] for i in range(n-1)]
    chunks.append(arr[(n-1)*size:])

    with multiprocessing.Pool() as pool:
        results = pool.map(chunk_stats, chunks)

    mins, maxs, sums = zip(*results)

    min_val = min(mins)
    max_val = max(maxs)
    total_sum = sum(sums)
    avg = total_sum / len(arr)

    print("Min:", min_val)
    print("Max:", max_val)
    print("Sum:", total_sum)
    print("Average:", avg)


if __name__ == "__main__":
    arr = [5, 2, 9, 1, 7, 6, 8, 3, 4]
    parallel_reduction(arr)