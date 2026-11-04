# 환경 세팅

처음 한 번만 하면 됩니다. 순서대로 따라 하세요.

## 0. 준비물

- Git
- Python 3.11
- [Google Drive 데스크톱](https://www.google.com/drive/download/)

## 1. 저장소 받기

```bash
git clone https://github.com/HanyangTechAI/2026-2-HAI-ProjectTeam2.git
cd 2026-2-HAI-ProjectTeam2
```

## 2. Python 패키지 설치

가상환경을 만들고 패키지를 설치합니다.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac
source .venv/bin/activate

pip install -r requirements.txt
```

## 3. 공유 드라이브를 PC에 연결

오디오 파일은 GitHub에 올리지 않고 공유 드라이브에 둡니다. 드라이브 링크는 Notion에 있습니다.

1. Google Drive 데스크톱을 실행하고 로그인합니다.
2. 웹의 **공유 문서함**에서 `HAI_2026-2_project2_drive` 옆 **⋮ → 정리 → 바로가기 추가 → 내 드라이브**를 누릅니다.
3. PC의 `내 드라이브`에 생긴 `HAI_2026-2_project2_drive` 바로가기를 더블클릭해 엽니다.
4. 탐색기(Finder) 주소창의 경로를 복사해 둡니다. 아래에서 `<드라이브 경로>`로 씁니다.
   - Windows 예: `G:\.shortcut-targets-by-id\<영문숫자>\HAI_2026-2_project2_drive`

## 4. 저장소와 드라이브 폴더 연결

코드는 저장소 안의 `data/...` 경로를 읽습니다. 이 경로가 드라이브 폴더를 가리키도록 연결합니다. 저장소 루트에서 실행하세요.

**Windows** (PowerShell 말고 **cmd**에서 실행)

```
mklink /J data\raw "<드라이브 경로>\data\raw"
mklink /J data\dry\Fender_Strat_Clean\audio "<드라이브 경로>\data\dry\Fender_Strat_Clean\audio"
mklink /J data\synthetic\Fender_Strat_Clean "<드라이브 경로>\data\synthetic\Fender_Strat_Clean"
```

**Mac**

```bash
DRIVE="<드라이브 경로>"
ln -s "$DRIVE/data/raw" data/raw
ln -s "$DRIVE/data/dry/Fender_Strat_Clean/audio" data/dry/Fender_Strat_Clean/audio
ln -s "$DRIVE/data/synthetic/Fender_Strat_Clean" data/synthetic/Fender_Strat_Clean
```

> "이미 있는 파일" 오류가 나면, 저장소에 같은 이름의 폴더가 이미 있는 것입니다. 그 폴더를 지우거나 이름을 바꾼 뒤 다시 실행하세요.

## 5. 확인

```bash
python schemas/validate.py
python -m unittest discover -s schemas -p "test_*.py"
python -m unittest discover -s metrics -p "test_*.py"
```

세 명령이 모두 통과하고, `data/dry/Fender_Strat_Clean/audio` 안에 wav 78개가 보이면 끝입니다.

## 지켜야 할 것

- **합성 데이터 생성은 한 사람만 합니다.** 드라이브에 바로 저장되므로, 여러 명이 동시에 돌리면 서로의 파일을 덮어씁니다.
- **드라이브를 외부에 공개하지 않습니다.** 데이터셋 라이선스(CC BY-NC-ND 4.0)상 이펙트를 건 음원은 재배포할 수 없습니다.
- **드라이브 링크를 저장소에 올리지 않습니다.** 링크는 Notion에서만 관리합니다.
- 처음 여는 파일은 내려받느라 느릴 수 있습니다. 자주 쓰는 폴더는 드라이브 앱에서 "오프라인 사용 가능"으로 설정하세요.
