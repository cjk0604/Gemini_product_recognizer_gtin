# Product Recognizer GTIN using Gemini

GTIN(Global Trade Item Number) 기반 상품 인식 시스템으로, 매장 선반 이미지에서 상품을 식별하여 카탈로그의 정확한 상품과 매칭합니다.

[Vision AI Product Recognizer](https://docs.cloud.google.com/vision-ai/docs/product-recognizer) 기능이 deprecate 될 예정이므로, 본 프로젝트는 [Multi-Modal Embedding](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-embeddings-api)과 VectorDB를 활용하여 상품 이미지 검색을 수행하고, Gemini를 통해 검색 후보 중 최적의 상품을 선택합니다. 전체 구성도는 아래 아키텍처를 참고하세요.

## 아키텍처

![alt text](high_level_architecture.png)

```
 ======================== 데이터 준비 (오프라인, 일회성) ==========================

  +------------+     +---------------------+     +-------------------------+
  | 상품       |     | 멀티모달            |     | Vertex AI               |
  | 카탈로그   |---->| 임베딩 모델         |---->| Matching Engine         |
  | (CSV)      |     | (이미지→128D 벡터)  |     | (Vector Index)          |
  +------------+     +---------------------+     +-------------------------+
       |                                                  |
       |         +--------------+                         |
       +-------->| GCS Bucket   |<------------------------+
                 | (이미지 &    |   임베딩 캐시 저장
                 |  임베딩 캐시)|
                 +--------------+

 ============================ 추론 (온라인, 쿼리 시) ===============================

  +------------------+
  | 매장 선반        |
  | 촬영 이미지      |
  +--------+---------+
           |
           v
  +----------------------------------+
  | Step 1. 이미지 전처리             |
  |                                  |
  |  +--------------+ +------------+ |
  |  | 정사각형     | | 마진       | |
  |  | 패딩         | | 크롭       | |
  |  | (흰색 배경)  | | (맥락 포함)| |
  |  +------+-------+ +-----+------+ |
  +---------|------------|---+--------+
            |            |
            v            |
  +-----------------------|----------+
  | Step 2. 벡터 검색     |          |
  |                       |          |
  |  패딩된 이미지        |          |
  |       |               |          |
  |       v               |          |
  |  임베딩 모델          |          |
  |       |               |          |
  |       v               |          |
  |  Matching Engine      |          |
  |  (ANN 검색)           |          |
  |       |               |          |
  |       v               |          |
  |  Top-K 후보           |          |
  |  (20개 상품)          |          |
  +---------+-------------|----------+
            |             |
            v             v
  +--------------------------------------+
  | Step 3. Gemini 리랭킹 (LLM)          |
  |                                      |
  |  입력:                               |
  |   - 쿼리 이미지 (패딩)              |
  |   - 줌아웃 이미지 (세그먼트)         |
  |   - 후보 상품 이미지 + 타이틀        |
  |                                      |
  |  Gemini 2.0 Flash                    |
  |  (브랜드/변형/무게/패키지            |
  |   시각적 비교)                       |
  |                                      |
  |  출력:                               |
  |   { color, title, candidate }        |
  +------------------+-------------------+
                     |
                     v
          +---------------------+
          | 매칭된 상품         |
          | (GTIN 식별 완료)    |
          +---------------------+
```

### 핵심 설계 결정

- **2단계 Retrieve & Rerank**: 벡터 검색으로 빠르게 후보를 좁히고(Step 2), Gemini LLM이 시각적으로 정밀 비교하여 최종 상품을 선택합니다(Step 3).
- **이중 쿼리 이미지**: 패딩(정사각형, 흰색 배경)과 마진 크롭(선반 맥락 포함) 이미지를 함께 사용하여 인식 정확도를 높입니다.
- **Punting (거부)**: 빈 선반, 가격표 오탐, 매칭 불가 시 `candidate = -1`을 반환하여 오인식을 방지합니다.
- **GCS 캐싱**: 이미지와 임베딩을 GCS에 캐시하여 반복 연산을 방지합니다.
- **GTIN 중복 제거**: 여러 가지 중복 제거 전략으로 동일 상품이 후보에 중복 노출되는 것을 방지합니다.

## 노트북 셀 구성

| 셀 | 제목 | 설명 |
|------|-------|-------------|
| 0 | Imports | 라이브러리 설치 및 임포트 (Google Cloud, Vertex AI, Gemini, PIL 등) |
| 1 | Global Parameters & Initialization | 프로젝트 ID, GCS 버킷, 임베딩 차원(128), 모델 버전, 재시도 전략 설정 |
| 2 | [Library] Vertex Matching Engine | 핵심 클래스: `GcsUri`(GCS 이미지 관리, 임베딩, 이웃 검색), `Product`(메타데이터), `Match`(검색 결과). 데이터셋 로딩, 인덱스 업서트, 시각화 함수 |
| 3 | [Library] Gemini Reranking | 6가지 Gemini 프롬프트 설정(기본, punting, letter, CoT 변형). 프롬프트 생성 함수(single/margin/letter/reverse). 응답 파싱 |
| 4 | Read Catalog | 상품 카탈로그 CSV를 읽어 GTIN→이미지/타이틀 딕셔너리 구축 |
| 5 | Prepare Vertex Matching Engine | Vector Index 및 IndexEndpoint 생성 또는 기존 리소스 재사용 |
| 6 | [RunOnce] Load data and build index | 상품 데이터셋 로딩, 임베딩 생성, Vector Index에 업서트 |
| 7 | [Optional][RunOnce] Pad product image | 상품 이미지를 정사각형으로 패딩(흰색 배경) - 쿼리 전처리 |
| 8 | [Optional][RunOnce] Segment product | 매장 프레임에서 상품 영역을 마진 포함하여 크롭 - 쿼리 전처리 |
| 9 | [Demo] End2End flow | 단일 상품에 대한 전체 파이프라인 데모 실행 |

## 사전 준비사항

### 1. Google Cloud 프로젝트

| 항목 | 설명 | 노트북 파라미터 |
|------|------|----------------|
| GCP 프로젝트 | Vertex AI가 활성화된 프로젝트 | `PROJECT_ID` |
| 리전 | 기본값: `us-central1` | `LOCATION` |
| GCS 버킷 | 이미지/임베딩 캐시 및 데이터셋 저장용 | `BUCKET` |
| 인증 | `gcloud auth login` 또는 서비스 계정 설정 완료 | - |

### 2. 활성화해야 할 GCP API

- **Vertex AI API** - Matching Engine, 임베딩 모델, Gemini
- **Cloud Storage API** - GCS 버킷 접근
- **Generative AI API** - Gemini 모델 호출

### 3. IAM 권한

실행 계정에 다음 역할이 필요합니다:

- **Vertex AI User** (`roles/aiplatform.user`) - Matching Engine, 임베딩 모델, Gemini 호출
- **Storage Object Admin** (`roles/storage.objectAdmin`) - GCS 읽기/쓰기 (이미지 캐싱)

### 4. 데이터 준비

#### 상품 카탈로그 CSV (`DATASET`)

GCS 버킷 내에 위치해야 합니다. 필수 컬럼:

```csv
entity_uuid,product_title,brand_name,gtins,images
abc-123,"Product Name","Brand",1234567890|9876543210,https://img1.jpg||https://img2.jpg
```

- `gtins`: 파이프(`|`)로 구분된 상품 바코드
- `images`: 이중 파이프(`||`)로 구분된 이미지 URL

#### 쿼리셋 JSONL (`QUERYSET`, 평가용)

각 행에 `imageUri` 필드를 포함하는 JSON 객체:

```json
{"imageUri": "https://example.com/shelf-photo.jpg"}
```

#### Detection 결과 CSV (선택, 이미지 전처리용)

필수 컬럼: `imageUri`, `frameUrl`, `x`, `y`, `w`, `h` (매장 프레임에서 탐지된 상품의 바운딩 박스)

### 5. 모델 선택

| 모델 | 파라미터 | 용도 |
|------|----------|------|
| 멀티모달 임베딩 | `EMBEDDING_MODEL_NAME` | 이미지를 128차원 벡터로 변환 |
| Gemini | `GEMINI_MODEL_VERSION` | LLM 리랭킹 (기본값: `gemini-2.0-flash-001`) |

### 6. Vertex AI Matching Engine 리소스

최초 실행 시 자동 생성되며, 기존 리소스 재사용도 가능합니다:

| 리소스 | 파라미터 | 비고 |
|--------|----------|------|
| Vector Index | `INDEX_NAME` | Tree-AH, 코사인 거리, 스트림 업데이트 |
| Index Endpoint | `INDEX_ENDPOINT_NAME` | **최초 배포 시 ~20분 소요** |
| Deployed Index ID | `DEPLOYED_INDEX_ID` | 배포된 인덱스 ID |

## 실행 순서

```
1. 파라미터 설정                 (Cell 0-1)
2. 라이브러리 초기화             (Cell 2-3)
3. 카탈로그 로딩                 (Cell 4)     <- 상품 CSV 필요
4. Matching Engine 준비          (Cell 5)     <- 최초 실행 시 ~20분
5. 인덱스 구축                   (Cell 6)     <- 일회성, 임베딩 생성 + 업서트
6. 이미지 전처리                 (Cell 7-8)   <- 선택, 일회성
7. 추론 실행                     (Cell 9)     <- 반복 가능
```

## 비용 고려사항

| 리소스 | 과금 |
|--------|------|
| Matching Engine Index Endpoint | 배포 중 시간당 과금 |
| Gemini API 호출 | 토큰 기반 과금 (쿼리당 이미지 여러 장 포함) |
| 임베딩 모델 호출 | 호출 기반 과금 (상품 수 x 이미지 수) |
| GCS 저장 및 전송 | 캐시된 이미지/임베딩 용량에 따라 과금 |

## Python 의존성

```
retrying
pandarallel
Pillow
ipywidgets
requests
rich
tqdm
google-cloud-aiplatform
google-cloud-storage
google-genai
```
