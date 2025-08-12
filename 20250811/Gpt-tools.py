import json
import os

DATA_FILE = 'chat_groups.json'

def load_data():
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def add_chat(group, chat_text):
    data = load_data()
    if group not in data:
        data[group] = []
    data[group].append(chat_text)
    save_data(data)
    print(f'[{group}] 그룹에 대화가 저장되었습니다.')

def get_chats(group=None):
    data = load_data()
    if group:
        return data.get(group, [])
    else:
        return data  # 전체 그룹과 대화 반환

# 사용 예시
if __name__ == "__main__":
    while True:
        print("\n1. 대화 저장하기\n2. 그룹별 대화 불러오기\n3. 전체 보기\n4. 종료")
        choice = input("선택: ").strip()
        
        if choice == '1':
            group = input("그룹명(예: 머신러닝, 부동산): ").strip()
            chat = input("대화 내용 입력: ").strip()
            add_chat(group, chat)
        
        elif choice == '2':
            group = input("불러올 그룹명 입력: ").strip()
            chats = get_chats(group)
            if chats:
                print(f"\n[{group}] 그룹 대화 목록:")
                for i, c in enumerate(chats, 1):
                    print(f"{i}. {c}")
            else:
                print("해당 그룹에 저장된 대화가 없습니다.")
        
        elif choice == '3':
            all_data = get_chats()
            print("\n전체 대화 목록:")
            for g, chats in all_data.items():
                print(f"\n[{g}] 그룹:")
                for i, c in enumerate(chats, 1):
                    print(f"  {i}. {c}")
        
        elif choice == '4':
            print("프로그램 종료.")
            break
        
        else:
            print("잘못된 입력입니다. 다시 시도해주세요.")
