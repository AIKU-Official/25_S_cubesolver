from argparse import ArgumentParser
import pickle

import numpy as np


def print_stats(data, hist=False):
    print("Min/Max/Median/Mean(Std) %f/%f/%f/%f(%f)" % (min(data), max(data), float(np.median(data)),
                                                        float(np.mean(data)), float(np.std(data))))
    if hist:
        hist1 = np.histogram(data)
        for x, y in zip(hist1[0], hist1[1]):
            print("%s %s" % (x, y))


def print_results(results):
    times = np.array(results["times"])
    lens = np.array([len(x) for x in results["solutions"]])
    num_nodes_generated = np.array(results["num_nodes_generated"])

    print("-Times-")
    print_stats(times)
    print("-Lengths-")
    print_stats(lens)
    print("-Nodes Generated-")
    print_stats(num_nodes_generated)
    print("-Nodes/Sec-")
    print_stats(np.array(num_nodes_generated) / np.array(times))


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--soln1', type=str, required=True, help="Ground truth file")
    parser.add_argument('--soln2', type=str, required=True, help="A* results file")

    args = parser.parse_args()

    results1 = pickle.load(open(args.soln1, "rb"))
    results2 = pickle.load(open(args.soln2, "rb"))

    if 'costs' in results1:
        lens1 = np.array(results1['costs'])
        print(f"Loaded optimal lengths from '{args.soln1}' using 'costs' key.")
    elif 'solutions' in results1:
        lens1 = np.array([len(x) for x in results1["solutions"]])
        print(f"Loaded optimal lengths from '{args.soln1}' using 'solutions' key.")
    else:
        raise KeyError(f"Could not find 'costs' or 'solutions' in {args.soln1}")

    # 두 번째 파일(soln2)은 항상 'solutions' 키를 가지고 있어야 합니다.
    if 'solutions' not in results2:
        raise KeyError(f"Could not find 'solutions' in {args.soln2}")
    lens2 = np.array([len(x) for x in results2["solutions"]])

    print(f"\nTotal puzzles: {len(lens1)}")

    # 첫 번째 파일은 통계 정보가 없으므로, 두 번째 파일의 상세 정보만 출력합니다.
    print(f"\n--- A* Search Results ({args.soln2}) ---")
    print_results(results2)

    # 두 결과의 해답 길이를 비교합니다.
    print("\n\n------ Comparison: (A* Lengths) - (Optimal Lengths) -----")
    print_stats(lens2 - lens1, hist=False)
    print(f"{100 * np.mean(lens2 == lens1):.2f}% of puzzles solved optimally.")


if __name__ == "__main__":
    main()
