# vast.ai GPU 세션 가이드 (한 번에 B1 · B2 · A)

`gpu_session.sh` 하나로 세 가지를 순서대로 돌린다.

| 단계 | 내용 | 대략 시간 (4090) |
|---|---|---|
| B1 | Qwen2.5-7B Study 1 재실행 + 사용자 단위 bootstrap (`user_bootstrap.json`) | ~1시간 |
| B2 | Llama-3.1-8B Study 1 재현 (`HF_TOKEN` 없으면 건너뜀) | ~1시간 |
| A | Study 3: 취향 프로필 × 글 쌍 → 내부 표상과 A/B 로짓이 실제 선택을 맞히나 | ~1–1.5시간 |

예상 비용: 4090 시간당 $0.3–0.5 × 3–4시간 ≈ $1–2.

## 0. 미리 (Mac에서, 한 번만)

1. Llama를 쓸 거면 https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct 에서 접근 승인을 받는다. 승인까지 몇 시간 걸릴 수 있다.
2. HF 토큰(read 권한)을 만든다. 토큰은 레포나 파일에 넣지 말고, 인스턴스에서 `export`만 한다.
3. 최신 코드가 GitHub에 push돼 있는지 확인한다 (`git log origin/main -1`).

## 1. 인스턴스 만들기 (vast.ai 웹)

- GPU: RTX 4090 또는 A6000 등 **24GB 이상** 1장
- 디스크: **80GB** (모델 두 개 + hidden state)
- 템플릿: PyTorch (CUDA 12.x) 이미지
- 신뢰도(Reliability) 높은 호스트, 다운로드 속도 500Mbps 이상이면 모델 받는 시간이 줄어든다.
- 만든 뒤 **Connect**에서 SSH 명령(`ssh -p <port> root@<host>`)을 복사해 둔다.

## 1b. VS Code로 접속하기 (Remote-SSH)

Remote-SSH 확장과 vast.ai 키(`~/.ssh/id_ed25519_vast`)는 이미 있다.

1. vast.ai **Account → SSH Keys**에 `~/.ssh/id_ed25519_vast.pub` 내용이 등록돼 있는지 확인한다. 없으면 인스턴스를 만들기 **전에** 추가한다.
   ```bash
   cat ~/.ssh/id_ed25519_vast.pub | pbcopy
   ```
2. `~/.ssh/config`에서 예전 인스턴스 항목(`Host 104.233...`, `Host 47.186...`)을 지우고, 이름이 고정된 항목 하나로 바꾼다. 인스턴스를 새로 만들 때마다 `HostName`과 `Port`만 고치면 된다. 둘 다 vast.ai **Connect** 창의 `ssh -p <port> root@<host>`에 나온다.
   ```
   Host vast
     HostName <host>
     Port <port>
     User root
     IdentityFile ~/.ssh/id_ed25519_vast
     ServerAliveInterval 30
   ```
3. VS Code: `Cmd+Shift+P` → **Remote-SSH: Connect to Host…** → `vast` → 플랫폼은 Linux를 고른다.
4. 접속되면 **Terminal → New Terminal**을 열고 아래 3단계의 명령을 그대로 입력한다. 폴더는 `/root/llm-personalization`을 연다 (**File → Open Folder**, clone한 다음).

> [!] VS Code 터미널은 창을 닫거나 Mac이 잠들면 같이 끊긴다. **반드시 `tmux` 안에서** `gpu_session.sh`를 돌린다.
> 다시 접속한 뒤 `tmux attach -t run`으로 붙으면 진행 상황이 그대로 있다.

**VS Code에서 파일 주고받기**
- 올리기: Mac Finder에서 `LiteraryTaste` 폴더를 VS Code 탐색기의 `llm-personalization/` 위로 끌어다 놓는다. 아래 2단계의 `scp` 대신 써도 된다.
- 받기: 끝난 뒤 탐색기에서 `results_small.tar.gz` 우클릭 → **Download…** → Mac의 레포 폴더에 저장하고 `tar xzf results_small.tar.gz`.

이 경우 2단계와 4단계는 건너뛰고, 3단계에서 `ssh` 대신 VS Code 터미널을 쓴다. 위의 `Host vast` 항목이 있으면 Mac 터미널 명령도 `ssh vast`, `scp -r LiteraryTaste vast:~/llm-personalization/`처럼 짧아진다.

## 2. 데이터 올리기 (Mac 터미널)

LiteraryTaste는 git에 없으니 직접 올린다 (19MB).

```bash
ssh -p <port> root@<host> "git clone https://github.com/jiwooroh/llm-personalization.git"
```

```bash
cd "/Users/lucyroh/Desktop/STUDY/Data Projects/llm-personalization" && scp -P <port> -r LiteraryTaste root@<host>:~/llm-personalization/
```

## 3. 실행 (인스턴스 안에서)

```bash
ssh -p <port> root@<host>
```

SSH가 끊겨도 계속 돌도록 `tmux` 안에서 실행한다.

```bash
tmux new -s run
```

```bash
cd ~/llm-personalization && export HF_TOKEN=hf_...   # Llama 안 쓰면 생략
```

```bash
bash gpu_session.sh 2>&1 | tee session.log
```

- 빠져나오기: `Ctrl-b` 다음 `d`. 다시 붙기: `tmux attach -t run`.
- 일부만 돌리기: `SKIP=B2 bash gpu_session.sh` (쉼표로 여러 개: `SKIP=B1,B2`).

### 확인할 출력

- B1 끝: `user_bootstrap.py`가 `LLM probe - verbal calibrated ... CI [...]`를 찍는다. 예전의 `scalar bootstrap` 문구가 **나오지 않아야** 사용자 단위 검정이다.
- A 진행: `[own] 4000/11488 ...` 같은 줄. 끝나면 `knows / says` 표와 `own - other` 검정이 나온다.
- 에러가 나면 `session.log` 마지막 30줄을 복사해서 가져오면 된다.

## 4. 결과 받기 (Mac 터미널)

```bash
cd "/Users/lucyroh/Desktop/STUDY/Data Projects/llm-personalization" && scp -P <port> root@<host>:~/llm-personalization/results_small.tar.gz . && tar xzf results_small.tar.gz
```

`runs/qwen7b/`, `runs/llama8b/`, `runs/literary_taste/joint_qwen7b/`가 생긴다 (hidden state `H_*.npy`는 제외, 수 GB라서).

## 5. 인스턴스 Destroy

vast.ai 웹 → Instances → **Destroy** (휴지통 아이콘). **Stop만 하면 디스크 저장료가 계속 나간다.** 결과 tar를 Mac에서 풀어 본 다음에 지운다.

## 결과 읽는 법 (Study 3)

`runs/literary_taste/joint_qwen7b/joint_summary.json`

| 키 | 뜻 |
|---|---|
| `acc_knows_at_ref` | 조건별 probe 정확도 (같은 레이어, own 기준 최적 레이어) |
| `acc_says` | 조건별 "A or B" 로짓 정확도 (학습 없음) |
| `tests` | 독자 단위 paired bootstrap. 핵심은 `knows own - other`, `says own - other` |
| `study2_reference` | Study 2의 population / +revealed 정확도 (비교 기준) |

미리 정한 해석 기준:

- **own ≈ other** → 모델이 프로필을 읽어도 선택엔 안 쓴다. Study 2 결론 유지.
- **own > other가 knows에서만 유의** → 아는데 말로는 못 쓴다. Study 1의 행동 버전, 가장 좋은 결과.
- **own − other < revealed 이득(+1.8%p)** → "말보다 행동"이 LLM 표상에서도 유지.
- **own < own_pos** → 부정 문장("I don't care about…")이 실제 선택 예측에서 손해.
