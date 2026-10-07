// 전역 풀 및 변수 정의
const colorPool = [
    { name: "빨강", hex: "#ff4757" }, { name: "파랑", hex: "#1e90ff" },
    { name: "초록", hex: "#2ed573" }, { name: "노랑", hex: "#ffa502" },
    { name: "보라", hex: "#9b59b6" }, { name: "핑크", hex: "#ff6b81" },
    { name: "하늘", hex: "#70a1ff" }, { name: "오렌지", hex: "#ff7f50" }
];

const emojiPool = ["🦊", "🐰", "🦁", "🐸", "🐵", "🐥", "🐼", "🐷"];
const numberPool = ["1", "2", "3", "4", "5", "6", "7", "8"];
const shapePool = ["▲", "■", "●", "◆", "★", "♣", "♥", "♠"];
const textPool = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ"];

let stage = 1;
let score = 0;
let combo = 0; 
let isReverseRule = false; 
let currentPatternType = "color"; 

let originalCards = []; 
let currentCards = [];  
let correctCardIndices = []; 

let timerInterval;
let gameActive = false;
let MEMORY_TIME = 2000; 

// [피드백 반영] 유저 행동 패턴 추적 (심리 낚시용)
let userClickHistory = [0, 0, 0, 0]; // 각 인덱스별 누른 횟수 저장
let currentRoomType = "normal"; // normal, bonus, psycho (심리방)

const containerEl = document.getElementById("game-container");
const gridEl = document.getElementById("card-grid");
const instructionEl = document.getElementById("instruction");
const stageEl = document.getElementById("stage-display");
const scoreEl = document.getElementById("score-display");
const timerBarEl = document.getElementById("timer-bar");
const overlayEl = document.getElementById("screen-overlay");
const startBtn = document.getElementById("start-btn");
const gameOverTitle = document.getElementById("game-over-title");
const descModal = document.getElementById("desc-modal");
const focusBadgeEl = document.getElementById("focus-badge");

// Fisher-Yates 셔플 알고리즘
function shuffle(array) {
    let currentIndex = array.length, randomIndex;
    while (currentIndex !== 0) {
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;
        [array[currentIndex], array[randomIndex]] = [array[randomIndex], array[currentIndex]];
    }
    return array;
}

function openModal() { descModal.style.display = "flex"; }
function closeModal() { descModal.style.display = "none"; }

function startGame() {
    stage = 1;
    score = 0;
    combo = 0;
    userClickHistory = [0, 0, 0, 0];
    gameActive = true;
    overlayEl.style.display = "none";
    focusBadgeEl.style.display = "none";
    nextStage();
}

// [피드백 반영] 성장감과 극복감을 조율하는 유연한 룸타입 결정 시스템
function determineRoomType(stg) {
    if (stg === 5 || stg === 10 || stg === 15) {
        return "bonus"; // 숨고르기 및 쾌감 보너스 방
    }
    // 6단계 이후부터 무작위 혹은 특정 주기별 유저 저격 '심리방' 등장
    if (stg >= 6 && (stg % 3 === 0 || Math.random() < 0.4)) {
        return "psycho"; 
    }
    return "normal";
}

function nextStage() {
    if (!gameActive) return;

    currentRoomType = determineRoomType(stage);

    let isFocusMode = (combo >= 3);
    if (isFocusMode) {
        focusBadgeEl.style.display = "block";
        focusBadgeEl.innerText = `🔥 FOCUS x${combo} (위험/보상증가)`;
    } else {
        focusBadgeEl.style.display = "none";
    }

    // 방 종류에 따른 상단 텍스트 및 연출 차별화
    if (currentRoomType === "bonus") {
        stageEl.innerText = `🎁 BONUS STAGE`;
        instructionEl.innerText = "🎉 보너스 방! 아무 생각 없이 틀린 것을 편하게 고르세요!";
        instructionEl.style.color = "#2ed573";
    } else if (currentRoomType === "psycho") {
        stageEl.innerText = `🧠 PSYCHO STAGE ${stage}`;
        instructionEl.innerText = "⚠️ 시스템이 당신의 심리를 읽고 있습니다...";
        instructionEl.style.color = "#9b59b6";
    } else {
        stageEl.innerText = `STAGE ${stage}`;
        instructionEl.innerText = "기억하세요!";
        instructionEl.style.color = "#ffa502";
    }

    scoreEl.innerText = `SCORE: ${score}`;
    timerBarEl.style.width = "100%";
    gridEl.classList.add("pe-none"); 

    // [피드백 반영] 카드 등장 시 모든 카드에 임펙트 애니메이션 적용
    const cards = document.querySelectorAll(".card");
    cards.forEach((card, i) => {
        card.className = "card"; 
        // 딜레이를 주어 순차적으로 팝업되는 연출
        card.style.animationDelay = `${i * 0.05}s`;
        card.classList.add("card-appear");
    });

    // 패턴 무작위 설정 (성장감과 패턴 학습 유도)
    const types = ["color"];
    if (stage >= 2) types.push("emoji");
    if (stage >= 4) types.push("number");
    if (stage >= 7) types.push("shape", "text");
    currentPatternType = types[Math.floor(Math.random() * types.length)];

    // 보너스 방은 무조건 편안한 이모지나 컬러로 고정
    if (currentRoomType === "bonus") currentPatternType = Math.random() < 0.5 ? "color" : "emoji";

    let currentPool = [...colorPool];
    if (currentPatternType === "emoji") currentPool = emojiPool.map(e => ({name: e, hex: "#fff"}));
    if (currentPatternType === "number") currentPool = numberPool.map(n => ({name: n, hex: "#fff"}));
    if (currentPatternType === "shape") currentPool = shapePool.map(s => ({name: s, hex: "#fff"}));
    if (currentPatternType === "text") currentPool = textPool.map(t => ({name: t, hex: "#fff"}));

    let shuffledPool = shuffle([...currentPool]);
    let selectedElements = shuffledPool.slice(0, 4);
    let freshElement = shuffledPool[4]; 

    originalCards = [];
    for (let i = 0; i < 4; i++) {
        originalCards.push({
            text: selectedElements[i].name,
            color: currentPatternType === "color" ? selectedElements[i].hex : "#ffffff",
            id: selectedElements[i].name 
        });
    }

    if (currentPatternType === "color" && stage >= 4 && currentRoomType !== "bonus") {
        let textColors = selectedElements.map(e => e.hex);
        textColors = shuffle(textColors);
        for (let i = 0; i < 4; i++) {
            originalCards[i].color = textColors[i];
        }
    }

    // 첫 카드 상태 렌더링
    setTimeout(() => {
        renderCards(originalCards);
    }, 50);

    // [피드백 반영] 반응속도 속이기 (기습 딜레이 변조)
    let dynamicBlankDelay = 1000; 
    if (currentRoomType === "psycho") {
        // 유저가 예측하지 못하도록 공백 대기시간을 0.4초 ~ 1.8초 사이로 교란
        dynamicBlankDelay = Math.floor(Math.random() * 1400) + 400;
    }

    let currentMemoryTime = isFocusMode ? Math.max(MEMORY_TIME - 400, 700) : MEMORY_TIME;
    if (currentRoomType === "bonus") currentMemoryTime = 2500; // 보너스방은 넉넉하게

    setTimeout(() => {
        cards.forEach(card => card.classList.remove("card-appear"));
        instructionEl.innerText = "비우세요...";
        instructionEl.style.color = "#888888";

        setTimeout(() => {
            setupQuestion(freshElement, isFocusMode);
        }, dynamicBlankDelay); 

    }, currentMemoryTime);
}

function setupQuestion(freshElement, isFocusMode) {
    if (!gameActive) return;

    // 규칙 반전 설정 (보너스 방은 무조건 일반 규칙)
    isReverseRule = false;
    if (currentRoomType !== "bonus" && stage >= 3 && Math.random() < 0.35) {
        isReverseRule = true;
        instructionEl.innerText = "★반전★ 안 바뀐 것(그대로인 것)을 누르세요!";
        instructionEl.style.color = "#00d2d3";
    } else {
        instructionEl.innerText = "틀린 것(처음과 다른 것)을 고르세요!";
        instructionEl.style.color = "#ff4757";
    }

    gridEl.classList.remove("pe-none"); 
    currentCards = JSON.parse(JSON.stringify(originalCards));

    // [피드백 반영] 유저 습관 읽기 및 위치 낚시 매커니즘
    let targetIdx = Math.floor(Math.random() * 4);
    
    if (currentRoomType === "psycho") {
        // 유저가 여태 가장 많이 눌렀던 최애(?) 자리를 찾음
        let maxClickCount = Math.max(...userClickHistory);
        let favoriteIndices = [];
        userClickHistory.forEach((count, idx) => {
            if (count === maxClickCount) favoriteIndices.push(idx);
        });
        let userFavoriteIdx = favoriteIndices[Math.floor(Math.random() * favoriteIndices.length)];

        // 유저가 무의식적으로 손이 자주 가는 위치에 '반전형 정답'을 배치하거나 정답에서 제외하여 습관 저격
        if (isReverseRule) {
            // 안 바뀐 걸 골라야 할 때: 자주 누르는 자리를 '바꿔버려서' 오답 유도
            targetIdx = userFavoriteIdx; 
        } else {
            // 바뀐 걸 골라야 할 때: 자주 누르는 자리를 '그대로 둠으로써' 낚시
            let safeIndices = [0, 1, 2, 3].filter(i => i !== userFavoriteIdx);
            targetIdx = safeIndices[Math.floor(Math.random() * safeIndices.length)];
        }
    }
    
    // 카드 교체 진행
    currentCards[targetIdx] = {
        text: freshElement.name,
        color: currentPatternType === "color" ? freshElement.hex : "#ffffff",
        id: freshElement.name
    };

    const cards = document.querySelectorAll(".card");

    // [피드백 반영] 가짜 힌트 및 페이크 쉐이크 교란 시각화
    if (stage >= 2 && currentRoomType !== "bonus") {
        if (currentRoomType === "psycho" && Math.random() < 0.6) {
            // 가짜 힌트: 전혀 뚱딴지같은 카드를 마구 흔들어서 시선 분산
            cards.forEach((card, idx) => {
                if(idx !== targetIdx && Math.random() < 0.7) card.classList.add("fake-shake");
            });
            instructionEl.innerText += " (눈조심!)";
        } else {
            let fakeTarget = Math.floor(Math.random() * 4);
            if (isFocusMode || stage >= 6) {
                cards.forEach((card, idx) => {
                    if(idx !== targetIdx) card.classList.add("fake-shake");
                });
            } else {
                if (fakeTarget !== targetIdx) {
                    cards[fakeTarget].classList.add("fake-shake");
                }
            }
        }
        setTimeout(() => {
            cards.forEach(card => card.classList.remove("fake-shake"));
        }, 250);
    }

    let originalIdOrder = originalCards.map(c => c.id);
    
    // 고난도 셔플 알고리즘 패턴 학습 유도
    if (stage >= 8 && currentRoomType !== "bonus") {
        currentCards = shuffle(currentCards);
    }

    if (stage >= 11 && currentPatternType === "color" && currentRoomType !== "bonus") {
        let colors = currentCards.map(c => c.color);
        colors = shuffle(colors);
        for(let i=0; i<4; i++) {
            currentCards[i].color = colors[i];
        }
    }

    // 카드 반짝임 후 문제 출제
    cards.forEach(card => card.classList.add("card-blink"));
    setTimeout(() => {
        renderCards(currentCards);
        cards.forEach(card => card.classList.remove("card-blink"));
    }, 150);

    // 정답 인덱스 연산
    correctCardIndices = [];
    currentCards.forEach((curCard, idx) => {
        let origIdx = originalIdOrder.indexOf(curCard.id);
        if (isReverseRule) {
            if (origIdx !== -1) {
                let origCard = originalCards[origIdx];
                if (origCard.text === curCard.text && origCard.color === curCard.color) {
                    correctCardIndices.push(idx);
                }
            }
        } else {
            if (origIdx === -1 || (stage >= 8 && idx !== origIdx) || originalCards[origIdx].color !== curCard.color) {
                if (curCard.id === freshElement.name) { 
                    correctCardIndices.push(idx);
                }
            }
        }
    });

    if (correctCardIndices.length === 0) {
        isReverseRule = false;
        instructionEl.innerText = "틀린 것(처음과 다른 것)을 고르세요!";
        instructionEl.style.color = "#ff4757";
        currentCards.forEach((c, idx) => {
            if (c.id === freshElement.name) correctCardIndices.push(idx);
        });
    }

    startTimer(isFocusMode);
}

function renderCards(cardsArray) {
    const cards = document.querySelectorAll(".card");
    cards.forEach((card, idx) => {
        card.innerText = cardsArray[idx].text;
        card.style.color = cardsArray[idx].color;
    });
}

function startTimer(isFocusMode) {
    clearInterval(timerInterval);
    // 스테이지가 갈수록 압박감을 주지만, 보너스방은 넉넉히 제공하여 극복감 부여
    let baseTime = 3200 - (stage * 80);
    if (currentRoomType === "bonus") baseTime = 4000;
    if (isFocusMode) baseTime *= 0.75; 

    let timeLeft = Math.max(baseTime, 900); 
    let totalTime = timeLeft;

    timerInterval = setInterval(() => {
        timeLeft -= 50;
        let widthPercent = (timeLeft / totalTime) * 100;
        timerBarEl.style.width = `${widthPercent}%`;

        if (timeLeft <= 0) {
            clearInterval(timerInterval);
            gameOver(currentRoomType === "psycho" ? "AI의 심리 압박에 타이머마저 보지 못했군요!" : "시간 초과! 뇌 정지가 오셨나요?");
        }
    }, 50);
}

function checkAnswer(clickedIndex) {
    if (!gameActive) return;
    clearInterval(timerInterval);

    // [피드백 반영] 유저가 클릭한 그리드 인덱스 위치 로깅 (빅데이터 수집)
    userClickHistory[clickedIndex]++;

    if (correctCardIndices.includes(clickedIndex)) {
        combo++;
        let bonus = (combo >= 3) ? combo * 20 : 0;
        if (currentRoomType === "bonus") bonus += 100; // 보너스 방 전용 대량 가산점

        score += (stage * 10) + bonus;

        containerEl.classList.add("shake-effect");
        
        let successTexts = ["PERFECT MEMORY!", "뇌지컬 폭발!", "대단한 몰입도군요!", "CLEAR!"];
        if (currentRoomType === "bonus") successTexts = ["🔋 기억력 완충 완료!", "보너스 점수 획득!"];

        if (combo >= 3) {
            instructionEl.innerText = `🔥 ${successTexts[stage % successTexts.length]} (x${combo})`;
        } else {
            instructionEl.innerText = successTexts[stage % successTexts.length];
        }
        instructionEl.style.color = "#2ed573";

        setTimeout(() => {
            containerEl.classList.remove("shake-effect");
            stage++;
            nextStage();
        }, 450);

    } else {
        let loseReason = "손가락이 정직하게 옳은 걸 눌러버렸습니다!";
        if (isReverseRule) loseReason = "그대로인 걸 고르라니까 홀린 듯 틀린 걸 눌렀네요!";
        if (currentRoomType === "psycho") loseReason = "철저하게 설계된 AI의 심리 유도 패턴에 낚이셨습니다!";
        
        gameOver(loseReason);
    }
}

function gameOver(reason) {
    gameActive = false;
    clearInterval(timerInterval);
    combo = 0;
    
    let nickname = "";
    let nicknameColor = "";

    if (stage <= 4) {
        nickname = "단기 기억상실증 금붕어 🐟";
        nicknameColor = "#ff4757";
    } else if (stage <= 8) {
        nickname = "방금 폰 어디 뒀는지 찾는 사람 📱";
        nicknameColor = "#ffa502";
    } else if (stage <= 12) {
        nickname = "심리 트릭을 견뎌낸 일반인 🧍";
        nicknameColor = "#1e90ff";
    } else if (stage <= 17) {
        nickname = "패턴을 통달한 인간 알파고 🧠";
        nicknameColor = "#2ed573";
    } else {
        nickname = "메타버스 두뇌 마스터 🤖";
        nicknameColor = "#b534ee";
    }
    
    gameOverTitle.innerText = "GAME OVER";
    gameOverTitle.style.color = "#ff4757";
    
    let reasonEl = document.getElementById("game-over-reason");
    if (!reasonEl) {
        reasonEl = document.createElement("p");
        reasonEl.id = "game-over-reason";
        reasonEl.style.fontSize = "1.2rem";
        reasonEl.style.marginBottom = "20px";
        reasonEl.style.textAlign = "center";
        reasonEl.style.lineHeight = "1.6";
        overlayEl.insertBefore(reasonEl, overlayEl.querySelector(".btn-group"));
    }
    
    reasonEl.innerHTML = `
        ${reason}<br>
        <span style='color: ${nicknameColor}; font-weight: bold; font-size: 1.4rem; display: inline-block; margin-top: 10px;'>
            [당신의 등급] ${nickname}
        </span>
        <br><br>
        <span style='color:#2ed573; font-weight:bold; font-size:1.5rem;'>
            최종 STAGE: ${stage}<br>
            최종 SCORE: ${score}
        </span>
    `;
    
    startBtn.innerText = "다시 도전";
    overlayEl.style.display = "flex";
}