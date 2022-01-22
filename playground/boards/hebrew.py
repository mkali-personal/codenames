from codenames.game import Board, Card, CardColor, words_to_random_board

HEBREW_WORDS = [
    "מטען",
    "עלילה",
    "ניצחון",
    "כבש",
    "יוגה",
    "צבי",
    "אף",
    "מפגש",
    "דק",
    "פרץ",
    "שלם",
    "אדם",
    "הרמוניה",
    "זכוכית",
    "חשמל",
    "מעטפת",
    "אנרגיה",
    "קברן",
    "נחת",
    "חייזר",
    "שיר",
    "מיליונר",
    "לפיד",
    "יקום",
    "דרור",
]
HEBREW_BOARD_1 = Board(
    [
        Card("חיים", color=CardColor.GRAY),
        Card("ערך", color=CardColor.BLACK),
        Card("מסוק", color=CardColor.BLUE),
        Card("שבוע", color=CardColor.GRAY),
        Card("רובוט", color=CardColor.RED),
        Card("פוטר", color=CardColor.GRAY),
        Card("אסור", color=CardColor.BLUE),
        Card("דינוזאור", color=CardColor.BLUE),
        Card("מחשב", color=CardColor.RED),
        Card("מעמד", color=CardColor.GRAY),
        Card("בעל", color=CardColor.RED),
        Card("פנים", color=CardColor.RED),
        Card("פרק", color=CardColor.RED),
        Card("גפילטע", color=CardColor.BLUE),
        Card("שונה", color=CardColor.RED),
        Card("שכר", color=CardColor.RED),
        Card("קפיץ", color=CardColor.BLUE),
        Card("תרסיס", color=CardColor.GRAY),
        Card("דגל", color=CardColor.GRAY),
        Card("חופשה", color=CardColor.BLUE),
        Card("מועדון", color=CardColor.RED),
        Card("ציון", color=CardColor.BLUE),
        Card("שק", color=CardColor.GRAY),
        Card("אקורדיון", color=CardColor.RED),
        Card("ילד", color=CardColor.BLUE),
    ]
)

HEBREW_BOARD_2 = words_to_random_board(words=HEBREW_WORDS, seed=1)
HEBREW_BOARD_3 = words_to_random_board(words=HEBREW_WORDS, seed=2)
