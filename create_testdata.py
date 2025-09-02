import pickle
import os
from environments.cube2 import Cube2  # 2x2 큐브 환경 import

def generate_cube2_test_data():
    """2x2 큐브 테스트 데이터를 생성하고 저장합니다."""

    # --- 설정 ---
    num_puzzles = 1000  # 생성할 테스트 문제 개수
    scramble_depth = 30  # 섞는 횟수 (훈련 시 back_max와 동일하게 설정)
    output_dir = "data/cube2/test"
    output_path = os.path.join(output_dir, "data_1.pkl")

    print("2x2 큐브 테스트 데이터 생성을 시작합니다...")

    # --- 폴더 생성 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"폴더 생성: {output_dir}")

    # --- 문제 생성 ---
    env = Cube2()
    # generate_states는 (상태 리스트, 섞은 횟수 리스트)를 반환합니다.
    states, costs = env.generate_states(num_puzzles, (scramble_depth, scramble_depth))

    print(f"'{scramble_depth}'번 섞은 문제 {len(states)}개 생성 완료.")

    # --- 파일 저장 ---
    # astar.py가 요구하는 딕셔너리 형태로 데이터를 저장합니다.
    test_data = {
        "states": states,
        "costs": costs
    }

    with open(output_path, 'wb') as f:
        pickle.dump(test_data, f)

    print(f"테스트 데이터를 성공적으로 저장했습니다: {output_path}")


if __name__ == "__main__":
    generate_cube2_test_data()