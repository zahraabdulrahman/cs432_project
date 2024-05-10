import random
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import time
from sklearn.ensemble import RandomForestClassifier

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = HandDetector(maxHands=1)

timer = 0
stateResult = False
startGame = False
scores = [0, 0]  # [AI, Player]
playerMovesHistory = []

# Initialize machine learning model
model = RandomForestClassifier(n_estimators=100, random_state=42)


def reset_game():
    global scores, stateResult, startGame, timer
    scores = [0, 0]
    stateResult = False
    startGame = False
    timer = 0
    playerMovesHistory.clear()  # Clear player's move history


def ai_move(player_move):
    global model

    if playerMovesHistory:  # If player has made moves
        X_train = [[move] for move in playerMovesHistory]
        y_train = [move % 3 + 1 for move in playerMovesHistory]  # AI plays the next move

        # Train the model
        model.fit(X_train, y_train)

        # Predict AI's move based on player's move
        ai_move = model.predict([[player_move]])[0]
    else:
        ai_move = random.randint(1, 3)  # Default to random move

    return ai_move



def on_window_close(event, x, y, flags, param):
    if event == cv2.WND_PROP_ASPECT_RATIO and flags == cv2.WINDOW_GUI_EXPANDED:
        # Window close event
        reset_game()
        cv2.destroyAllWindows()


cv2.namedWindow("Rock, Paper, Scissors", cv2.WINDOW_GUI_EXPANDED)
cv2.setWindowProperty("Rock, Paper, Scissors", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_GUI_EXPANDED)
cv2.setWindowProperty("Rock, Paper, Scissors", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_GUI_EXPANDED)
cv2.setWindowProperty("Rock, Paper, Scissors", cv2.WND_PROP_TOPMOST, cv2.WINDOW_GUI_EXPANDED)
cv2.setMouseCallback("Rock, Paper, Scissors", on_window_close)

while True:
    imgBG = cv2.imread("Resources/BG.png")
    success, img = cap.read()

    imgScaled = cv2.resize(img, (0, 0), None, 0.875, 0.875)
    imgScaled = imgScaled[:, 80:480]

    # Find Hands
    hands, img = detector.findHands(imgScaled)  # with draw

    if startGame:
        if stateResult is False:
            timer = time.time() - initialTime
            cv2.putText(imgBG, str(int(timer)), (605, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)

            if timer > 3:
                stateResult = True
                timer = 0

                if hands:
                    hand = hands[0]
                    fingers = detector.fingersUp(hand)
                    if fingers == [0, 0, 0, 0, 0]:
                        playerMove = 1
                    elif fingers == [1, 1, 1, 1, 1]:
                        playerMove = 2
                    elif fingers == [0, 1, 1, 0, 0]:
                        playerMove = 3

                    aiMove = ai_move(playerMove)
                    imgAI = cv2.imread(f'Resources/{aiMove}.png', cv2.IMREAD_UNCHANGED)
                    imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

                    # Player Wins
                    if (playerMove == 1 and aiMove == 3) or \
                            (playerMove == 2 and aiMove == 1) or \
                            (playerMove == 3 and aiMove == 2):
                        scores[1] += 1

                    # AI Wins
                    if (playerMove == 3 and aiMove == 1) or \
                            (playerMove == 1 and aiMove == 2) or \
                            (playerMove == 2 and aiMove == 3):
                        scores[0] += 1

                    playerMovesHistory.append(playerMove)  # Addplayer's move to history

    imgBG[234:654, 795:1195] = imgScaled

    if stateResult:
        imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

    cv2.putText(imgBG, str(scores[0]), (410, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
    cv2.putText(imgBG, str(scores[1]), (1112, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)

    cv2.imshow("Rock, Paper, Scissors", imgBG)

    key = cv2.waitKey(1)
    if key == ord('s'):
        startGame = True
        initialTime = time.time()
        stateResult = False
    elif key == ord('r'):
        reset_game()

    if cv2.getWindowProperty("Rock, Paper, Scissors", cv2.WND_PROP_VISIBLE) < 1:
        # Window close event
        reset_game()
        break

cap.release()
cv2.destroyAllWindows()