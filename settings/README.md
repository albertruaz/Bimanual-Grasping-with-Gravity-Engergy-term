# Environment Settings

이 폴더에는 프로젝트 환경 복원을 위한 설정 파일들이 저장되어 있습니다.

## 파일 설명

### 1. `pip_list.txt`

- **설명**: `pip list` 명령어의 출력 결과
- **용도**: 설치된 Python 패키지들의 정확한 버전 정보 확인
- **생성 명령어**: `pip list > pip_list.txt`

### 2. `isaacgym_explicit.txt`

- **설명**: conda 환경의 explicit 설정 파일
- **용도**: conda 환경의 정확한 복원
- **생성 명령어**: `conda list --explicit > isaacgym_explicit.txt`

## 환경 복원 방법

### Conda 환경 복원 (권장)

```bash
# 새로운 환경 생성
conda create --name isaacgym --file settings/isaacgym_explicit.txt

# 환경 활성화
conda activate isaacgym
```

### pip 패키지 확인

환경 복원 후 pip_list.txt와 비교하여 누락된 패키지가 있는지 확인:

```bash
pip list
```

## 주요 종속성

### 핵심 패키지들

- **Python**: 3.7.16
- **PyTorch**: 1.12.1 (CUDA 11.6)
- **PyTorch3D**: 0.7.1
- **IsaacGym**: 1.0rc4
- **CUDA Toolkit**: 11.6.0

### 프로젝트별 패키지들

- **pytorch-kinematics**: 0.3.0 (local editable install)
- **torchsdf**: 0.1.0 (local editable install)
- **transforms3d**: 0.4.2
- **trimesh**: 4.5.3

## 주의사항

1. **IsaacGym 설치**: IsaacGym은 별도 설치가 필요합니다
2. **CUDA 호환성**: CUDA 11.6과 호환되는 GPU 드라이버가 필요합니다
3. **로컬 패키지**: pytorch-kinematics와 torchsdf는 로컬 editable install입니다

## 문제 해결

환경 복원 시 문제가 발생하면:

1. CUDA 버전 호환성 확인
2. IsaacGym 라이센스 및 설치 확인
3. 로컬 패키지 경로 확인
