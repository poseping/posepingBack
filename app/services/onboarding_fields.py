# 최대 채팅 턴
ONBOARDING_MAX_TURNS = 10

# 채팅에서 수집할 정보
ONBOARDING_FIELDS = [
    {
        "key": "sitting_hours_per_day",
        "label": "하루에 앉아있는 시간",
        "description": "하루 평균 몇 시간 앉아 있는지",
        "question": "하루에 얼마나 앉아 계신가요?",
    },
    {
        "key": "exercise_days_per_week",
        "label": "일주일에 운동하는 일수",
        "description": "일주일 기준 운동하는 날 수",
        "question": "일주일에 몇 번 운동하세요?",
    },
    {
        "key": "pain_areas",
        "label": "통증이 있다면 통증 부위",
        "description": "통증이 없으면 없음으로 기록",
        "question": "불편하거나 아픈 부위가 있나요?",
    },
    {
        "key": "sleep_position",
        "label": "수면 시 자세",
        "description": "",
        "question": "잘 땐 주로 어떤 자세로 누워있나요?"
    }
]
