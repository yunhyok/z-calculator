# Z Calculator

PCB 전송선의 단면 치수로 특성 임피던스와 손실을 계산하는 Windows 데스크톱 앱. 목표 임피던스에서 도선 하부 선폭을 역산하고, 같은 조건으로 다시 해석한 결과를 표시한다.

**[Windows x64 설치 파일 다운로드](https://github.com/yunhyok/z-calculator/releases/latest)** · 기본 설치 폴더 `C:\Program Files\Z Calculator`. 설치 화면에서 다른 드라이브와 폴더 선택 가능.

![차동 스트립라인 100 Ω 설계 화면](assets/workspace.png)

## 01 / 사용 순서

1. **단면**에서 Microstrip, Stripline, Coupled microstrip, Coupled stripline 중 하나를 선택할 것. 도선 하부 선폭, 두께, 접지면까지의 거리 입력. 차동쌍은 도선 사이 간격도 입력.
2. **재료**에서 로컬 Cadence 재료 라이브러리(`.cmx`)를 열고 유전체 선택. 직접 입력 시 상대 유전율 Dk와 손실 탄젠트 Df 지정. 앱이 마지막 라이브러리 경로를 기억한다.
3. **해석 조건**에서 주파수 범위, 설계 주파수, 선로 길이, 양단 포트 임피던스 설정.
4. **임피던스 계산** 또는 **목표 선폭 계산** 실행. 차동 구조의 목표와 포트 임피던스는 차동값. 역산 시 간격·두께·측벽 각도·접지 실드 간격은 고정.
5. **주파수 응답** 목록에서 그래프 선택 후 **PNG 저장**. **단면 PNG**는 치수선이 포함된 단면 개념도 저장. **PowerPoint 보고서**는 조건·재료·단면·전체 그래프·대표 수치·모델 가정을 담은 `.pptx` 생성.

## 02 / 단면과 재료

| 입력 | 정의 |
| --- | --- |
| W / T | 도선 하부 선폭 / 구리 두께. 길이 단위 µm |
| 측벽 각도 θ | 기판 면 기준 45°–90°. 기본 90°. 상부 선폭 = W − 2T / tan θ |
| H₁ / H₂ | 구리 하부에서 아래 접지면까지 / 구리 상부에서 위 접지면까지 거리 |
| S / G | 차동쌍의 하부 가장자리 간격 / 바깥 도선과 동일층 접지 실드 사이 하부 간격 |
| Microstrip | 하부 유전체와 접지면, 상부 공기. 솔더마스크 없는 노출 도선 |
| Stripline | 상·하부 접지면. 양측 유전체와 높이를 독립 지정. 도선 옆의 재료 경계는 구리 두께 중앙 |
| 접지 실드 | 신호선과 같은 층에 놓인 양측 접지 도체. 실드의 접지 연결은 이상적으로 모델링 |

CMX 라이브러리는 사용자 PC에서 읽는다. 주파수 표는 로그 주파수에서 Dk·Df를 보간하고, 범위 밖은 가장 가까운 끝값을 사용한다. 단일 기준점과 수동 입력은 주파수 전체에 상수로 적용한다. 온도 표기가 없는 기본 모델을 우선 사용하며, 온도별 모델만 있을 때는 25 °C에 가장 가까운 모델을 선택한다.

## 03 / 결과 해석

| 결과 | 단위 | 의미 |
| --- | --- | --- |
| 특성 임피던스 | Ω | 손실을 포함한 복소 특성 임피던스의 실수부 Re(Zc). 차동쌍은 2 × Zodd |
| 단위 길이당 DCR | Ω/m | 20 °C 구리 도선 한 가닥의 단면적 기준 직류 저항. 주파수에 대해 일정 |
| 반사 손실 RL | dB | 지정한 길이의 균일 선로를 동일한 두 실수 기준 임피던스에 연결한 −20 log₁₀ abs(S11). 클수록 반사가 작음 |
| 단위 길이당 전파 손실 | dB/m | 전파 상수 실수부의 감쇠. 정합된 선로의 단위 길이당 삽입 손실 |
| 지정 길이 삽입 손실 IL | dB | −20 log₁₀ abs(S21). 선로 길이와 양단 불정합 포함 |

<strong>제작 판단에는 실제 적층 물성과 측정 또는 독립 전자기장 해석으로 교차 확인할 것.</strong> 이 모델은 균일한 직선 구간의 준정적 TEM 근사다. 매끄러운 구리, 이상적으로 연결된 기준면, 등방성 유전체를 가정한다. 동박 거칠기, 도금, 비아·커넥터 전이, 복사 및 고차 모드는 계산 범위 밖이다. 반사가 사라지는 조건에서는 그래프의 반사 손실을 최대 300 dB로 표시한다.

## 04 / 계산 방법과 검증

- 도체 경계에 맞춘 삼각형 유한요소 격자로 단면 전위를 푼다. 진공과 유전체의 정전용량으로 외부 인덕턴스를 산출하고, 재료별 전계 에너지와 표면 전계로 유전체 및 도체 손실을 추정한다. 준정적 전송선 관계는 Clayton R. Paul의 교재 참고. [Analysis of Multiconductor Transmission Lines, 2nd edition](https://www.wiley.com/en-us/analysis-of-multiconductor-transmission-lines-2nd-edition-p-9780470131541)
- 얇은 도체의 Microstrip에서 W/H = 0.2, 0.5, 1, 2, 5, 10과 Dk = 4로 준정적 임피던스를 비교한다. 이 검증의 허용차는 2%이며 다른 구조나 손실 정확도에 그대로 적용할 수 없다. [Hammerstad & Jensen, Accurate Models for Microstrip Computer-Aided Design](https://doi.org/10.1109/MWSYM.1980.1124303)
- 균일 유전체 Stripline의 스케일링, 넓은 평행판 극한, 차동 간격·실드·경사 각의 변화 방향, ABCD와 S파라미터의 일치, 네 구조의 선폭 역산을 자동 검사한다. UI 테스트는 재료 선택, 설계 조건 변경, PNG 생성과 PPTX 재열기를 포함한다.

## 05 / 소스 실행과 설치 파일 빌드

Windows x64, Python 3.12, PowerShell 환경. 설치 파일 빌드에는 Inno Setup 6.3 이상과 한국어 언어 파일 필요.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python main.py
```

```powershell
.\.venv\Scripts\python -m pip install pyinstaller==6.22.3 pytest
powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\build.ps1
```

빌드 결과는 `dist\Z-Calculator-1.0.0-Windows-x64-Setup.exe`. 배포 구성 요소와 라이선스는 설치 폴더의 `_internal\licenses`에 포함된다.
