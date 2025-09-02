# 프로젝트명 : Cube Solver

📢 2025년 여름학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다

## 소개
어려운 큐브, 딥러닝으로 풀어보자! 큐브를 푸는 Agent만들기!

## 방법론
- DeepCubeA
    - Model 구조
        - projection (cube_state to hidden) → 2 FC (feature extraction) → 4 Residual Block → output layer
        - state → model → value
    - Deep Approximate Value Iteration (DAVI)
        - DNN으로 Value Estimation
    - Batch Weighted A* Search
        - DAVI로 얻은 cost-to-go function을 heuristic으로 사용하여 Cost Tree에서 A* search 진행하여 다음 action을 정함
        - Iteration마다 N 개의 lowest cost nodes를 batch로 parallel하게 계산 가능!

## 환경 설정

Python version used: 3.7.2

IMPORTANT! Before running anything, please execute: source setup.sh in the DeepCubeA directory to add the current directory to your python path.

## 사용 방법
훈련 코드
`python ctg_approx/avi.py --env cube2 --states_per_update 50000000 --batch_size 10000 --nnet_name cube2 --max_itrs 1000000 --loss_thresh 0.1 --back_max 500 --num_update_procs 30`

모델 성능 측정 코드
`python search_methods/astar.py --states data/cube2/test/data_0.pkl --model saved_models/cube2/current/ --env cube2 --weight 0.8 --batch_size 20000 --results_dir results/cube2/ --language cpp --nnet_batch_size 10000`

1회 inference 코드
`python scripts/solve_one_cube2.py --model_dir saved_models/cube2/current --scramble_len 20 --batch_size 64 --weight 1.0`

## 예시 결과
1회 inference 결과
![예시 결과 사진](https://github.com/AIKU-Official/25_S_cubesolver/blob/master/Result_Image.png?raw=true)

## 팀원

- [고건영] : 시각화 및, 데모 코드 작성, 발표 자효 제작
- [박찬우] : DeepCubeA 모델 코드 작성
- [박보건] : DeepCube 코드 실험, 수행결과, 노션 정리
