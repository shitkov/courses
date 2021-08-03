import requests
import json

def get_id(user_id):
    
    ACCESS_TOKEN = '17da724517da724517da72458517b8abce117da17da72454d235c274f1a2be5f45ee711'

    r1 = requests.get(
        'https://api.vk.com/method/users.get?v=5.71&access_token=' + 
        ACCESS_TOKEN +
        '&user_ids=' +
        user_id
        )

    return r1.json()['response'][0]['id']


def get_friends(user_id):

    ACCESS_TOKEN = '17da724517da724517da72458517b8abce117da17da72454d235c274f1a2be5f45ee711'

    r1 = requests.get(
        'https://api.vk.com/method/friends.get?v=5.71&access_token=' + 
        ACCESS_TOKEN +
        '&user_id=' +
        str(user_id) +
        '&fields=bdate'
        )

    return r1.json()['response']['items']


def get_bdays(friends):
    ans = []
    for f in friends:
        try:
            if len(f['bdate']) >= 8:
                ans.append(2020 - int(f['bdate'][-4:]))
        except:
            pass
    return ans


def get_ans(age):
    dct = {}
    for i in set(age):
        dct[str(i)] = 0

    for i in age:
        dct[str(i)] = dct[str(i)] + 1
    
    ans = [(int(list(dct.keys())[i]), list(dct.values())[i]) for i in range(len(dct))]
    
    return sorted(sorted(ans, key = lambda x : x[0]), key = lambda x : x[1], reverse = True)


def calc_age(uid):
    id = get_id(uid)
    friends = get_friends(id)
    age = get_bdays(friends)
    ans = get_ans(age)
    return ans


if __name__ == '__main__':
    res = calc_age('reigning')
    print(res)
