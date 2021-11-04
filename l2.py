import numpy as np
from math import factorial
from dataclasses import dataclass
from prettytable import PrettyTable
import matplotlib.pyplot as plt


n = 3
m = 2
lyambda = 5
mu = 4
v = 3
MAX_TIME = 1000

CURRENT_TIME = 0
beta = v / mu
t_ozh = 1 / v
ro = lyambda / mu
requests, queue, smo = None, None, None


@dataclass
class Values:
    p: list    # Финальные вероятности
    p_otk: float   # Вероятность отказа
    A: float    # Ср ч заявок в единицу времени/Абсолютная пропускная способность
    L_och: float    # Среднее число заявок в очереди
    t_smo: float    # Среднее время пребывания в СМО
    t_och: float    # Среднее время пребывания заявки в очереди
    L_obc: float     # Среднее число каналов в СМО, занятых обслуживанием заявок
    L_smo: float    # Среднее число заявок, находящихся в СМО


def generate_p():
    p = []

    p0 = sum([(ro ** k) / factorial(k) for k in range(0, n + 1)])
    summa = 0
    for i in range(1, m + 1):
        summa += ro ** i / np.prod([n + l * beta for l in range(1, i + 1)])
    p0 += summa * ((ro ** n) / factorial(n))
    p0 = 1 / p0
    p.append(p0)

    for k in range(1, n + 1):
        p.append(p0 * (ro ** k) / factorial(k))

    for i in range(1, m + 1):
        p_n_i = p[n] * (ro ** i / np.prod([n + l * beta for l in range(1, i + 1)]))
        p.append(p_n_i)

    return p


def get_theor_values():
    p = generate_p()
    p_otk = p[-1]
    Q = 1 - p_otk
    A = lyambda * Q
    L_och = sum([i * p[n + i] for i in range(1, m + 1)])
    L_obc = sum([k * p[k] for k in range(1, n + 1)]) + sum([n * p[n + i] for i in range(1, m + 1)])
    t_och = L_och / lyambda
    L_smo = sum([i * p[i] for i in range(len(p))])
    t_smo = Q / mu + t_och

    return Values(p, p_otk, A, L_och, t_smo, t_och, L_obc, L_smo)


def generate_requests():
    r = []
    t = 0
    while t < MAX_TIME:
        t += np.random.exponential(1 / lyambda)
        r.append(t)
    return r


def get_next_item():
    global requests, queue, smo

    min_request = min(requests)
    min_queue = None if len(queue) == 0 else min(queue)
    min_smo = None if len(smo) == 0 else min(smo)

    min_of_min = min([q for q in [min_request, min_queue, min_smo] if q is not None])
    if min_of_min == min_request:
        return 'request', min_of_min
    if min_of_min == min_queue:
        return 'queue', min_of_min
    if min_of_min == min_smo:
        return 'smo', min_of_min
    raise ValueError('смэрть')


def main():
    global CURRENT_TIME, MAX_TIME, requests, queue, smo
    requests = generate_requests()  # Время захода
    queue = []  # (Время ухода, Время прихода)
    smo = []  # Время ухода

    empir_p = [0 for _ in range(n + m + 1)]
    request_count = len(requests)
    unmanaged_request_count = 0

    while CURRENT_TIME < MAX_TIME:
        event_name, time = get_next_item()

        if event_name == 'queue':   # Истекло время нахождения в очереди
            empir_p[len(smo) + len(queue)] += time - CURRENT_TIME
            queue.remove(time)

        if event_name == 'smo':
            empir_p[len(smo) + len(queue)] += time - CURRENT_TIME
            smo.remove(time)    # Освобождаем обслугу

            if len(queue) != 0:
                queue.pop(0)
                delta = np.random.exponential(1 / mu)
                smo.append(time + delta)    # Берем из очереди

        if event_name == 'request':
            empir_p[len(smo) + len(queue)] += time - CURRENT_TIME

            requests.remove(time)
            if len(smo) < n:
                delta = np.random.exponential(1 / mu)
                smo.append(time + delta)     # Если есть место на обслугу, встаем на нее
            elif len(queue) < m:
                queue.append(time + np.random.exponential(1 / v))   # Если есть место в очереди, встаем в нее
            else:
                unmanaged_request_count += 1
                pass    # Уходим если занято все


        CURRENT_TIME = time

    return get_empir_values(empir_p, request_count, unmanaged_request_count)


def get_empir_values(empir_p, request_count, unmanaged_request_count):
    empir_A = (request_count - unmanaged_request_count) / MAX_TIME
    normalized_empir_p = [e_p / MAX_TIME for e_p in empir_p]
    empir_p_otk = normalized_empir_p[-1]
    empir_L_och = sum([i * normalized_empir_p[n + i] for i in range(1, m + 1)])
    empir_L_obc = sum([k * normalized_empir_p[k] for k in range(1, n + 1)]) + sum([n * normalized_empir_p[n + i] for i in range(1, m + 1)])
    empir_t_och = empir_L_och / (empir_A / (1 - normalized_empir_p[-1]))
    empir_L_smo = sum([i * normalized_empir_p[i] for i in range(len(normalized_empir_p))])
    empir_t_smo = (1 - normalized_empir_p[-1]) / mu + empir_t_och

    return Values(normalized_empir_p, empir_p_otk, empir_A, empir_L_och, empir_t_smo, empir_t_och, empir_L_obc, empir_L_smo)


def show_results(theor, empir):
    print(f'n = {n}, m = {m}, λ = {lyambda}, μ = {mu}, v = {v}, Временной интервал: {MAX_TIME}\n')

    print('Финальные вероятности состояний')
    th = [i for i in range(n + m + 1)]
    th.insert(0, 'Эмп/Теор')
    table = PrettyTable(th)
    table.add_row(['Теор', *theor.p])
    table.add_row(['Эмпир', *empir.p])
    print(table)

    print(f'Вероятность отказа - теор: {theor.p_otk}, эмпир: {empir.p_otk}')
    print(f'Абсолютная пропускная способность - теор: {theor.A}, эмпир: {empir.A}')
    print(f'Среднее число заявок в очереди - теор: {theor.L_och}, эмпир: {empir.L_och}')
    print(f'Среднее время пребывания в СМО - теор: {theor.t_smo}, эмпир: {empir.t_smo}')
    print(f'Среднее время пребывания заявки в очереди - теор: {theor.t_och}, эмпир: {empir.t_och}')
    print(f'Среднее число каналов в СМО, занятых обслуживанием заявок - теор: {theor.L_obc}, эмпир: {empir.L_obc}')
    print(f'Среднее число заявок, находящихся в СМО - теор: {theor.L_smo}, эмпир: {empir.L_smo}')

    s = [60 for _ in range(n + m + 1)]
    x = [i for i in range(n + m + 1)]
    plt.scatter(x, theor.p, s=s, c='red')
    plt.plot(x, theor.p, c='red')
    plt.scatter(x, empir.p, s=s, c='blue')
    plt.plot(x, empir.p, c='blue')
    plt.show()


if __name__ == '__main__':
    theor_values = get_theor_values()
    empir_values = main()
    show_results(theor_values, empir_values)

