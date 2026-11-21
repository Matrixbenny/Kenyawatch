"""Third synthetic corpus: 10,000 reports in the four ways Kenyans write.

Kenyans report the same incident in very different ways. Some write plain,
careful English; some write plain Kiswahili; many mix English and Kiswahili in
one sentence; and many, especially young people in towns, write Sheng. This
generator builds 2,500 fictional scenarios from everyday Kenyan settings
(traffic roadblocks, county askari, the lands registry, Huduma Centres, county
hospitals, Kenya Power, the chief's office, KRA at the port, school
admissions, bursary and CDF money, county tenders and jobs) and writes each
scenario once in each register:

    English    "Yesterday, a traffic police officer at the roadblock near Nakuru
                asked me for KSh 500 through M-Pesa before he would let our matatu go."
    Kiswahili  "Jana, afisa wa polisi wa trafiki kwenye kizuizi karibu na Nakuru
                aliniomba shilingi mia tano kupitia M-Pesa ndipo aruhusu matatu yetu iende."
    Mixed      "Jana traffic cop pale roadblock karibu na Nakuru aliniambia nitoe
                500 bob kwa M-Pesa ndio aachilie matatu yetu iende."
    Sheng      "Jana karao wa traffic pale roadblock ya Nakuru alinidai soo tano
                kwa M-Pesa ndio aachilie mat yetu."

The four versions of a scenario describe the same incident (same setting,
official, amount, time and closing remark) and share a group_id, so they always
fall in the same validation fold. Every scenario is fictional; nothing here is
an observation of a real report. Because the reports come from templates, a
model can learn the templates, so this corpus must be judged on how much it
helps on text it was not generated from, not on its own cross-validation score.

    python -m ml.src.generate_register_dataset
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

from ml.src.dataset import CATEGORIES

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"
OUT_CSV = OUT_DIR / "kenya_watch_synthetic_registers.csv"
OUT_META = OUT_DIR / "kenya_watch_synthetic_registers.metadata.json"
SCENARIOS = 2500
SEED = 20261121
REGISTERS = ("English", "Kiswahili", "Mixed", "Sheng")
TOWNS = ("Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Machakos", "Kakamega", "Nyeri", "Garissa",
         "Kitui", "Kilifi", "Bungoma", "Meru", "Kisii", "Embu", "Narok", "Thika", "Malindi", "Kericho", "Homa Bay")

# ---------------------------------------------------------------------------
# Money, written the way each register writes it.
# ---------------------------------------------------------------------------
NUM = {1: "moja", 2: "mbili", 3: "tatu", 4: "nne", 5: "tano", 6: "sita", 7: "saba", 8: "nane", 9: "tisa",
       10: "kumi", 20: "ishirini", 50: "hamsini"}


def _sw_words(amount: int) -> str | None:
    """Kiswahili number words: mia tano, elfu kumi, laki mbili, milioni moja."""
    if amount < 1000 and amount % 100 == 0 and amount // 100 in NUM:
        return f"mia {NUM[amount // 100]}"
    if amount < 100_000 and amount % 1000 == 0 and amount // 1000 in NUM:
        return f"elfu {NUM[amount // 1000]}"
    if amount < 1_000_000 and amount % 100_000 == 0 and amount // 100_000 in NUM:
        return f"laki {NUM[amount // 100_000]}"
    if amount % 1_000_000 == 0 and amount // 1_000_000 in NUM:
        return f"milioni {NUM[amount // 1_000_000]}"
    return None


def _short(amount: int) -> str:
    if amount >= 1_000_000:
        return f"{amount / 1_000_000:g}M"
    if amount >= 1000:
        return f"{amount // 1000}k"
    return f"{amount} bob"


def money(amount: int, register: int, rng: random.Random) -> str:
    if register == 0:
        if amount >= 1_000_000:
            return rng.choice([f"KSh {amount / 1_000_000:g} million", f"KSh {amount:,}"])
        return rng.choice([f"KSh {amount:,}", f"{amount:,} shillings", f"Sh{amount:,}", f"Ksh. {amount:,}"])
    words = _sw_words(amount)
    if register == 1:
        options = [f"shilingi {amount:,}", f"Sh{amount:,}"]
        if words:
            options += [f"shilingi {words}"] * 2
        elif amount >= 1_000_000:
            options.append(f"shilingi milioni {amount / 1_000_000:g}")
        return rng.choice(options)
    if register == 2:
        options = [f"{amount:,} bob", f"Ksh {amount:,}", _short(amount)]
        if amount >= 100_000 and words:
            options.append(words)
        return rng.choice(options)
    sheng = []
    if amount == 200:
        sheng = ["mbao kumi", "soo mbili"]
    elif amount < 1000 and amount // 100 in NUM:
        sheng = [f"soo {NUM[amount // 100]}"]
    elif amount == 1000:
        sheng = ["thao", "ngiri"]
    elif amount < 100_000 and amount % 1000 == 0 and amount // 1000 in NUM:
        sheng = [f"thao {NUM[amount // 1000]}", f"ngiri {NUM[amount // 1000]}"]
    elif amount < 1_000_000 and words:
        sheng = [words]
    elif amount >= 1_000_000:
        sheng = [f"milli {NUM[amount // 1_000_000]}" if amount % 1_000_000 == 0 and amount // 1_000_000 in NUM
                 else f"{amount / 1_000_000:g} milli"]
    options = sheng * 2 + [_short(amount)] + ([f"{amount} bob"] if amount <= 5000 else [])
    return rng.choice(options)


# ---------------------------------------------------------------------------
# Slot pools. Every tuple is (English, Kiswahili, Mixed, Sheng). {town} is
# filled per scenario. Settings give the official, where they work, the
# service asked for as a subjunctive clause ("aruhusu ...", used after
# ndipo/ndio) and as an infinitive ("kuruhusu ...", used after alikataa).
# ---------------------------------------------------------------------------
SETTINGS = [
    dict(off=("a traffic police officer", "afisa wa polisi wa trafiki", "traffic cop", "karao wa traffic"),
         place=("at the roadblock near {town}", "kwenye kizuizi karibu na {town}", "pale roadblock karibu na {town}", "pale roadblock ya {town}"),
         svc=("let our matatu go", "aruhusu matatu yetu iende", "aachilie matatu yetu iende", "aachilie mat yetu"),
         inf=("let our matatu go", "kuruhusu matatu yetu iende", "kuachilia matatu yetu", "kuachilia mat yetu")),
    dict(off=("a county askari", "askari wa kaunti", "askari wa county", "kanjo"),
         place=("in {town} market", "katika soko la {town}", "pale market ya {town}", "pale soko ya {town}"),
         svc=("let me keep selling at my stall", "aniruhusu niendelee kuuza kibandani", "aniachie niendelee ku-sell kwa stall yangu", "aniachilie niendelee kuuza kwa kibanda"),
         inf=("let me keep selling at my stall", "kuniruhusu niendelee kuuza kibandani", "kuniachia niendelee ku-sell kwa stall yangu", "kuniachilia niendelee kuuza kwa kibanda")),
    dict(off=("a clerk at the lands registry", "karani wa ofisi ya ardhi", "clerk wa lands office", "msee wa lands"),
         place=("in {town}", "huko {town}", "pale {town}", "pale {town}"),
         svc=("process my title deed", "ashughulikie hati yangu ya shamba", "a-process title deed yangu", "anipitishie title yangu"),
         inf=("process my title deed", "kushughulikia hati yangu ya shamba", "ku-process title deed yangu", "kunipitishia title yangu")),
    dict(off=("a nurse at {town} County Referral Hospital", "muuguzi wa hospitali ya rufaa ya {town}", "nurse wa {town} county hospital", "nesi wa hosi ya county {town}"),
         place=("", "", "", ""),
         svc=("admit my sick mother", "amlaze mama yangu mgonjwa", "a-admit mama yangu ambaye ni mgonjwa", "amlaze mathe yangu mgonjwa"),
         inf=("admit my sick mother", "kumlaza mama yangu mgonjwa", "ku-admit mama yangu ambaye ni mgonjwa", "kumlaza mathe yangu mgonjwa")),
    dict(off=("an officer at the {town} Huduma Centre", "afisa wa Huduma Centre ya {town}", "officer wa Huduma Centre {town}", "msee wa Huduma {town}"),
         place=("", "", "", ""),
         svc=("speed up my ID card", "aharakishe kitambulisho changu", "a-speed up ID yangu", "aharakishe ID yangu"),
         inf=("speed up my ID card", "kuharakisha kitambulisho changu", "ku-speed up ID yangu", "kuharakisha ID yangu")),
    dict(off=("a Kenya Power technician", "fundi wa Kenya Power", "technician wa KPLC", "fundi wa stima"),
         place=("in {town}", "huko {town}", "huko {town}", "huko {town}"),
         svc=("connect electricity to my house", "aunganishe stima nyumbani kwangu", "a-connect stima kwa nyumba yangu", "aniwekee stima kwa keja"),
         inf=("connect electricity to my house", "kuunganisha stima nyumbani kwangu", "ku-connect stima kwa nyumba yangu", "kuniwekea stima kwa keja")),
    dict(off=("the assistant chief", "chifu msaidizi", "assistant chief", "chifu"),
         place=("of our location in {town}", "wa kata yetu huko {town}", "wa location yetu {town}", "wa location yetu {town}"),
         svc=("sign my bursary form", "atie sahihi fomu yangu ya basari", "a-sign form yangu ya bursary", "anisainie form ya bursary"),
         inf=("sign my bursary form", "kutia sahihi fomu yangu ya basari", "ku-sign form yangu ya bursary", "kunisainia form ya bursary")),
    dict(off=("an immigration officer", "afisa wa uhamiaji", "officer wa immigration", "msee wa immigration"),
         place=("at Nyayo House", "katika Nyayo House", "pale Nyayo House", "pale Nyayo House"),
         svc=("release my passport", "aniachilie pasipoti yangu", "a-release passport yangu", "aniachilie passport"),
         inf=("release my passport", "kuniachilia pasipoti yangu", "ku-release passport yangu", "kuniachilia passport")),
    dict(off=("an officer at {town} police station", "afisa wa kituo cha polisi cha {town}", "cop pale {town} police station", "karao wa stesheni ya {town}"),
         place=("", "", "", ""),
         svc=("give me a police abstract", "anipe abstract ya polisi", "anipe police abstract", "anipe abstract"),
         inf=("give me a police abstract", "kunipa abstract ya polisi", "kunipa police abstract", "kunipa abstract")),
    dict(off=("an officer at {town} police station", "afisa wa kituo cha polisi cha {town}", "cop pale {town} police station", "karao wa stesheni ya {town}"),
         place=("", "", "", ""),
         svc=("release my brother from the cells", "amwachilie kaka yangu kutoka seli", "a-release brother yangu from the cells", "amwachilie bro wangu kutoka cell"),
         inf=("release my brother from the cells", "kumwachilia kaka yangu kutoka seli", "ku-release brother yangu from the cells", "kumwachilia bro wangu kutoka cell")),
    dict(off=("a county public health officer", "afisa wa afya wa kaunti", "health officer wa county", "msee wa health wa county"),
         place=("in {town}", "huko {town}", "pale {town}", "pale {town}"),
         svc=("approve the licence for my food kiosk", "aidhinishe leseni ya kibanda changu cha chakula", "a-approve licence ya food kiosk yangu", "anipitishie licence ya kibanda ya chakula"),
         inf=("approve the licence for my food kiosk", "kuidhinisha leseni ya kibanda changu cha chakula", "ku-approve licence ya food kiosk yangu", "kunipitishia licence ya kibanda ya chakula")),
    dict(off=("a clerk at the {town} law courts", "karani wa mahakama ya {town}", "clerk wa {town} law courts", "clerk wa korti ya {town}"),
         place=("", "", "", ""),
         svc=("find my case file", "atafute faili ya kesi yangu", "a-trace file ya case yangu", "anitafutie file ya case"),
         inf=("find my case file", "kutafuta faili ya kesi yangu", "ku-trace file ya case yangu", "kunitafutia file ya case")),
    dict(off=("an NTSA examiner", "mkaguzi wa NTSA", "examiner wa NTSA", "msee wa NTSA"),
         place=("in {town}", "huko {town}", "pale {town}", "pale {town}"),
         svc=("pass me in the driving test", "anipitishe kwenye mtihani wa udereva", "anipitishe kwa driving test", "anipitishe driving test"),
         inf=("pass me in the driving test", "kunipitisha kwenye mtihani wa udereva", "kunipitisha kwa driving test", "kunipitisha driving test")),
    dict(off=("a KRA officer at the port", "afisa wa KRA bandarini", "KRA officer pale port", "msee wa KRA pale port"),
         place=("in Mombasa", "Mombasa", "ya Mombasa", "ya Mombasa"),
         svc=("clear my container", "aachilie kontena langu", "a-clear container yangu", "aniclearie container"),
         inf=("clear my container", "kuachilia kontena langu", "ku-clear container yangu", "kuniclearia container")),
    dict(off=("the principal of {town} Boys High School", "mkuu wa shule ya upili ya wavulana ya {town}", "principal wa {town} Boys", "mwalimu mkuu wa {town} Boys"),
         place=("", "", "", ""),
         svc=("admit my son to Form One", "amkubali mwanangu kujiunga na kidato cha kwanza", "a-admit son wangu Form One", "amchukue mtoi wangu Form One"),
         inf=("admit my son to Form One", "kumkubali mwanangu kujiunga na kidato cha kwanza", "ku-admit son wangu Form One", "kumchukua mtoi wangu Form One")),
    dict(off=("a clerk at the SHA office", "karani wa ofisi ya SHA", "clerk wa SHA", "msee wa SHA"),
         place=("in {town}", "huko {town}", "pale {town}", "pale {town}"),
         svc=("approve my hospital bill claim", "aidhinishe madai ya bili yangu ya hospitali", "a-approve claim ya hospital bill yangu", "anipitishie claim ya bill ya hosi"),
         inf=("approve my hospital bill claim", "kuidhinisha madai ya bili yangu ya hospitali", "ku-approve claim ya hospital bill yangu", "kunipitishia claim ya bill ya hosi")),
    dict(off=("an officer at the DCI", "afisa wa DCI", "officer wa DCI", "msee wa DCI"),
         place=("in {town}", "huko {town}", "pale {town}", "pale {town}"),
         svc=("issue my certificate of good conduct", "anipe cheti cha tabia njema", "anipe good conduct certificate", "anipe cert ya good conduct"),
         inf=("issue my certificate of good conduct", "kunipa cheti cha tabia njema", "kunipa good conduct certificate", "kunipa cert ya good conduct")),
    dict(off=("a county revenue officer", "afisa wa mapato wa kaunti", "revenue officer wa county", "msee wa revenue wa county"),
         place=("in {town}", "huko {town}", "pale {town}", "pale {town}"),
         svc=("issue my business permit", "anipe kibali changu cha biashara", "anipe business permit yangu", "anipe permit ya biashara"),
         inf=("issue my business permit", "kunipa kibali changu cha biashara", "kunipa business permit yangu", "kunipa permit ya biashara")),
    dict(off=("an officer at the weighbridge", "afisa wa mizani", "officer wa weighbridge", "msee wa weighbridge"),
         place=("near {town}", "karibu na {town}", "karibu na {town}", "karibu na {town}"),
         svc=("let my overloaded lorry pass", "aruhusu lori langu lililozidisha mzigo lipite", "aachilie lorry yangu ipite hata kama ni overloaded", "aachilie lorry yangu ipite"),
         inf=("let my overloaded lorry pass", "kuruhusu lori langu lililozidisha mzigo lipite", "kuachilia lorry yangu ipite hata kama ni overloaded", "kuachilia lorry yangu ipite")),
]

OFFICIALS = [
    ("the ward administrator", "msimamizi wa wadi", "ward administrator", "ward admin"),
    ("our MCA", "MCA wetu", "MCA wetu", "MCA wetu"),
    ("the county finance officer", "afisa wa fedha wa kaunti", "finance officer wa county", "msee wa finance wa county"),
    ("the school principal", "mkuu wa shule", "principal wa shule", "mwalimu mkuu"),
    ("the CDF committee chairman", "mwenyekiti wa kamati ya CDF", "chairman wa CDF committee", "chair wa CDF"),
    ("the hospital administrator", "msimamizi wa hospitali", "hospital administrator", "admin wa hosi"),
    ("the sub-county administrator", "msimamizi wa kaunti ndogo", "sub-county administrator", "sub-county admin"),
    ("the county chief officer for water", "afisa mkuu wa maji wa kaunti", "chief officer wa water", "CO wa maji"),
    ("the area chief", "chifu wa eneo letu", "area chief", "chifu wa area"),
    ("the procurement officer", "afisa wa ununuzi", "procurement officer", "msee wa procurement"),
]
PROJECTS = [  # Kiswahili always "fedha za ..." (zi- agreement); Mixed and Sheng "pesa/doo ya ..." (i- agreement).
    ("the bursary money for needy students", "fedha za basari za wanafunzi maskini", "pesa ya bursary ya wanafunzi", "doo ya bursary"),
    ("the CDF money for the borehole", "fedha za CDF za kisima", "pesa ya CDF ya borehole", "doo ya CDF ya borehole"),
    ("the money for repairing our road", "fedha za kukarabati barabara yetu", "pesa ya ku-repair road yetu", "ganji ya kurepair barabara"),
    ("the hospital drugs budget", "fedha za kununua dawa za hospitali", "pesa ya madawa ya hospital", "doo ya madawa ya hosi"),
    ("the money for relief food", "fedha za chakula cha msaada", "pesa ya relief food", "doo ya relief food"),
    ("the school feeding money", "fedha za chakula cha shule", "pesa ya school feeding", "doo ya chakula ya shule"),
    ("the youth fund money", "fedha za mfuko wa vijana", "pesa ya youth fund", "doo ya youth fund"),
    ("the market fees collected every week", "fedha za ada za soko", "pesa ya market fees", "doo ya market"),
    ("the money for the new classrooms", "fedha za madarasa mapya", "pesa ya classrooms mpya", "doo ya madarasa mpya"),
    ("the women's group loan", "fedha za mkopo wa kikundi cha wanawake", "pesa ya loan ya women group", "doo ya loan ya chama ya mamas"),
    ("the money for the dispensary", "fedha za zahanati", "pesa ya dispensary", "doo ya dispensary"),
    ("the Inua Jamii money for the elderly", "fedha za Inua Jamii za wazee", "pesa ya Inua Jamii ya wazee", "doo ya Inua Jamii ya wazee"),
]
ITEMS = [
    ("school desks", "madawati ya shule", "desks za shule", "madesk za shule"),
    ("hospital beds", "vitanda vya hospitali", "beds za hospital", "vitanda za hosi"),
    ("road gravel", "changarawe ya barabara", "gravel ya road", "gravel ya barabara"),
    ("laptops for the county offices", "kompyuta za ofisi za kaunti", "laptops za county offices", "laptops za county"),
    ("fertiliser for farmers", "mbolea ya wakulima", "fertiliser ya farmers", "fertiliser ya wakulima"),
    ("water tanks", "matangi ya maji", "water tanks", "matank za maji"),
    ("ambulances", "magari ya wagonjwa", "ambulances", "ambulance"),
    ("street lights", "taa za barabarani", "street lights", "taa za mtaa"),
    ("textbooks", "vitabu vya shule", "textbooks", "vitabu za shule"),
    ("medical supplies", "vifaa vya matibabu", "medical supplies", "vifaa za hosi"),
]
FIRMS = [
    ("a company registered only last month", "kampuni iliyosajiliwa mwezi uliopita tu", "company iliyo-registriwa last month tu", "kampuni ilisajiliwa mwezi jana tu"),
    ("a company with no office", "kampuni isiyo na ofisi", "company ambayo haina office", "kampuni haina hata ofisi"),
    ("a briefcase company", "kampuni ya mfukoni", "briefcase company", "kampuni ya briefcase"),
    ("a company that has never supplied anything", "kampuni ambayo haijawahi kusambaza chochote", "company ambayo haija-supply kitu", "kampuni haijawahi supply kitu"),
]
RELATIVES = [  # Kiswahili nouns all take "wa".
    ("nephew", "mpwa", "nephew", "mpwa"),
    ("cousin", "binamu", "cousin", "binamu"),
    ("wife", "mke", "wife", "bibi"),
    ("brother-in-law", "shemeji", "brother-in-law", "shemeji"),
    ("son", "mtoto", "son", "mtoi"),
    ("girlfriend", "mpenzi", "girlfriend", "manzi"),
    ("close friend", "rafiki", "close friend", "beshte"),
    ("daughter", "binti", "daughter", "dame"),
]
OPPORTUNITIES = [  # Kiswahili nouns all take i- agreement (kazi, nafasi).
    ("the county job", "kazi ya kaunti", "job ya county", "job ya county"),
    ("the police recruitment slot", "nafasi ya kuajiriwa polisi", "slot ya police recruitment", "slot ya kuingia polisi"),
    ("the teaching post", "nafasi ya ualimu", "teaching job", "job ya ualimu"),
    ("the internship at the ministry", "nafasi ya mafunzo wizarani", "internship ya ministry", "internship ya ministry"),
    ("the scholarship", "nafasi ya ufadhili wa masomo", "scholarship", "scholarship"),
    ("the casual job on the road project", "kazi ya kibarua kwenye mradi wa barabara", "casual job ya road project", "kibarua ya road project"),
    ("the nursing vacancy at the hospital", "nafasi ya muuguzi hospitalini", "nurse vacancy ya hospital", "job ya nesi pale hosi"),
    ("the KDF recruitment slot", "nafasi ya kujiunga na KDF", "KDF recruitment slot", "slot ya KDF"),
]
DOCUMENTS = [  # (document, the office that issues it)
    (("KCSE certificate", "cheti cha KCSE", "KCSE certificate", "cheti ya KCSE"),
     ("the county education office", "ofisi ya elimu ya kaunti", "education office ya county", "ofisi ya education ya county")),
    (("title deed", "hati ya shamba", "title deed", "title"),
     ("the lands office in {town}", "ofisi ya ardhi ya {town}", "lands office ya {town}", "ofisi ya lands {town}")),
    (("birth certificate", "cheti cha kuzaliwa", "birth certificate", "birth cert"),
     ("the civil registration office in {town}", "ofisi ya usajili wa vizazi ya {town}", "civil registration office ya {town}", "ofisi ya birth cert {town}")),
    (("degree certificate", "cheti cha digrii", "degree certificate", "cheti ya degree"),
     ("the university registry", "masjala ya chuo kikuu", "registry ya university", "registry ya campus")),
    (("national ID", "kitambulisho", "ID", "ID"),
     ("the registration of persons office", "ofisi ya usajili wa watu", "registration office", "ofisi ya registration")),
    (("car logbook", "kitabu cha gari", "logbook ya gari", "logbook"),
     ("the NTSA office in {town}", "ofisi ya NTSA ya {town}", "NTSA office ya {town}", "ofisi ya NTSA {town}")),
    (("certificate of good conduct", "cheti cha tabia njema", "good conduct certificate", "cert ya good conduct"),
     ("the DCI office", "ofisi ya DCI", "DCI office", "ofisi ya DCI")),
    (("land rates clearance", "cheti cha kulipa ada ya ardhi", "land rates clearance", "clearance ya rates"),
     ("the county revenue office", "ofisi ya mapato ya kaunti", "revenue office ya county", "ofisi ya revenue ya county")),
]
JOB_DOCUMENTS = (0, 3, 6)        # documents someone could fake to get a job
IDENTITY_DOCUMENTS = (2, 4)      # documents that could be sold to a foreigner
OFFICES = [  # Kiswahili nouns all take i- agreement (ofisi, hospitali, masjala).
    ("the lands office in {town}", "ofisi ya ardhi ya {town}", "lands office ya {town}", "ofisi ya lands {town}"),
    ("the {town} Huduma Centre", "Huduma Centre ya {town}", "Huduma Centre {town}", "Huduma {town}"),
    ("the chief's office", "ofisi ya chifu", "office ya chief", "ofisi ya chifu"),
    ("the county hospital in {town}", "hospitali ya kaunti ya {town}", "county hospital ya {town}", "hosi ya county {town}"),
    ("the registry at the law courts", "masjala ya mahakama", "registry ya court", "registry ya korti"),
    ("the SHA office in {town}", "ofisi ya SHA ya {town}", "SHA office ya {town}", "ofisi ya SHA {town}"),
    ("the county revenue office", "ofisi ya mapato ya kaunti", "revenue office ya county", "ofisi ya revenue ya county"),
    ("the registration of persons office", "ofisi ya usajili wa watu", "registration office", "ofisi ya registration"),
]
PAYMENT = [("", "", "", ""), (" in cash", " taslimu", " cash", " cash"),
           (" through M-Pesa", " kupitia M-Pesa", " kwa M-Pesa", " kwa M-Pesa"),
           (" on M-Pesa to a different number", " kwa M-Pesa kwa namba nyingine", " kwa M-Pesa namba ingine", " kwa Mpesa namba ingine")]
PRONOUNS = (dict(he="he", him="him", his="his"), dict(he="she", him="her", his="her"))

# Time and place openings. "{t}" in a template accepts either; other templates
# (present-tense ones) accept only a place opening.
TIMES = [("Yesterday, ", "Jana, ", "Jana ", "Jana "), ("Last week, ", "Wiki iliyopita, ", "Last week ", "Wiki jana "),
         ("On Monday morning, ", "Jumatatu asubuhi, ", "Monday morning ", "Monday asubuhi "),
         ("This morning, ", "Leo asubuhi, ", "Leo asubuhi ", "Leo morning "), ("Last month, ", "Mwezi uliopita, ", "Last month ", "Mwezi jana ")]
PLACES = [("In {town}, ", "Huko {town}, ", "Huku {town} ", "Huku {town} ")]
INTERJECTIONS = ([], ["Ndugu, ", "Tafadhali, "], ["Aki ", "Guys, ", "Please help, "], ["Maze, ", "Aki ", "Wueh, ", "Si mnisaidie, "])
CLOSINGS = [
    (" Please investigate.", " Tafadhali chunguzeni.", " Please mchunguze hii.", " Mchunguze hii kitu tafadhali."),
    (" This has been going on for months.", " Jambo hili limekuwa likiendelea kwa miezi.", " Hii imekuwa ikiendelea for months.", " Hii kitu imekuwa ikiendelea miezi kibao."),
    (" Many people in our area know about it.", " Watu wengi eneo letu wanajua.", " Watu wengi area yetu wanajua.", " Wasee wengi mtaa wanajua."),
    (" I am afraid of being victimised for reporting this.", " Naogopa kudhulumiwa kwa kuripoti.", " Naogopa ku-victimisiwa juu ya ku-report.", " Naogopa watanitafuta juu ya kuripoti hii."),
    (" Kindly act on this.", " Tafadhali chukueni hatua.", " Kindly chukueni action.", " Mchukue action tafadhali."),
]
MONEY_CLOSINGS = [
    (" I have the M-Pesa message as proof.", " Nina ujumbe wa M-Pesa kama ushahidi.", " Niko na M-Pesa message as proof.", " Niko na message ya M-Pesa kama proof."),
    (" Everyone who goes there is asked for the same.", " Kila anayefika pale huombwa vivyo hivyo.", " Kila mtu anafika pale anaambiwa the same.", " Kila msee anafika pale anaitishwa hivo hivo."),
]

# ---------------------------------------------------------------------------
# Incidents. Each entry is one incident written four ways, so the versions of a
# scenario always describe the same thing. English-only slots: {he} {him} {his}.
# ---------------------------------------------------------------------------
T = {
    "Bribery": [
        ("{t}{off} {place} asked me for {amt}{pay} before {he} would {svc}.",
         "{t}{off} {place} aliniomba {amt}{pay} ndipo {svc}.",
         "{t}{off} {place} aliniambia nitoe {amt}{pay} ndio {svc}.",
         "{t}{off} {place} alinidai {amt}{pay} ndio {svc}."),
        ("{t}{off} {place} refused to {inf} until I paid {amt}{pay}.",
         "{t}{off} {place} alikataa {inf} mpaka nilipe {amt}{pay}.",
         "{t}{off} {place} alikataa {inf} mpaka nimpe {amt}{pay}.",
         "{t}{off} {place} alikataa {inf} mpaka nimtolee {amt}{pay}."),
        ("{t}I went to {off} {place} and was told nothing would happen unless I gave {him} {amt}.",
         "{t}nilienda kwa {off} {place} nikaambiwa hakuna kitakachofanyika nisipompa {amt}.",
         "{t}nilienda kwa {off} {place} akaniambia bila {amt} hakuna service.",
         "{t}nilienda kwa {off} {place} akaniambia bila kitu kidogo ya {amt} hakuna kitu."),
        ("{t}{off} {place} wanted {amt} on top of the official fee to {svc}.",
         "{t}{off} {place} alitaka {amt} zaidi ya ada rasmi ili {svc}.",
         "{t}{off} {place} alitaka {amt} extra juu ya official fee ndio {svc}.",
         "{t}{off} {place} aliitisha {amt} juu ya ile fee ya kawaida ndio {svc}."),
        ("{t}{off} {place} asked for 'something small', {amt}{pay}, to {svc}.",
         "{t}{off} {place} aliomba 'kitu kidogo' cha {amt}{pay} ili {svc}.",
         "{t}{off} {place} aliomba kitu kidogo, {amt}{pay}, ndio {svc}.",
         "{t}{off} {place} aliitisha kakitu kadogo, {amt}{pay}, ndio {svc}."),
        ("{t}when I asked {off} {place} to {svc}, {he} said I should 'add something' of {amt}.",
         "{t}nilipomwomba {off} {place} {svc}, alisema niongeze kitu cha {amt}.",
         "{t}nilipomwambia {off} {place} {svc}, akasema niongeze kitu {amt}.",
         "{t}nilipomwambia {off} {place} {svc}, akasema niongeze kitu {amt}."),
        ("{t}I had to pay {off} {place} {amt}{pay} just to {svc}. There was no receipt.",
         "{t}ilibidi nimlipe {off} {place} {amt}{pay} ili {svc}. Hakukuwa na risiti.",
         "{t}ilibidi nilipe {off} {place} {amt}{pay} ndio {svc}, na hakuna receipt.",
         "{t}ilibidi nimtolee {off} {place} {amt}{pay} ndio {svc}, hakuna risiti."),
    ],
    "Extortion": [
        ("{t}{off} {place} threatened to arrest me unless I paid {amt}{pay}.",
         "{t}{off} {place} alitishia kunikamata nisipolipa {amt}{pay}.",
         "{t}{off} {place} alinithreaten kuniarrest nisipolipa {amt}{pay}.",
         "{t}{off} {place} alinitisha atanishika nisipomwachia {amt}{pay}."),
        ("{t}{off} {place} said {he} would charge me with a made-up offence if I did not give {him} {amt}.",
         "{t}{off} {place} alisema atanishtaki kwa kosa la kubuni nisipompa {amt}.",
         "{t}{off} {place} alisema atani-charge na case ya uongo kama sitampa {amt}.",
         "{t}{off} {place} alisema atanibambikia case feki kama sitamtolea {amt}."),
        ("{t}{off} {place} took my phone and would only return it after I paid {amt}.",
         "{t}{off} {place} alichukua simu yangu na akasema ataitoa tu nikilipa {amt}.",
         "{t}{off} {place} alichukua phone yangu akasema atarudisha tu nikilipa {amt}.",
         "{t}{off} {place} alinyakua simu yangu akasema ataniregeshea tu nikimwachia {amt}."),
        ("{t}I was forced to pay {amt}{pay} to {off} {place} after {he} threatened to beat me.",
         "{t}nililazimishwa kumlipa {off} {place} {amt}{pay} baada ya kutishiwa kupigwa.",
         "{t}nili-force-iwa kulipa {off} {place} {amt}{pay} after akanithreaten kunipiga.",
         "{t}nililazimishwa kumwachia {off} {place} {amt}{pay} juu alinitisha atanichapa."),
        ("{t}{off} {place} locked me in the cells overnight and demanded {amt} for my release.",
         "{t}{off} {place} alinifungia seli usiku kucha na kudai {amt} ili aniachilie.",
         "{t}{off} {place} alini-lock kwa cell usiku mzima akadai {amt} ndio aniachilie.",
         "{t}{off} {place} alinifungia cell usiku mzima akaitisha {amt} ndio aniachilie."),
        ("{t}{off} {place} told me that if I did not pay {amt} every week, my business would be closed.",
         "{t}{off} {place} aliniambia nisipolipa {amt} kila wiki biashara yangu itafungwa.",
         "{t}{off} {place} aliniambia nisipolipa {amt} kila week, biashara yangu itafungwa.",
         "{t}{off} {place} aliniambia nisipotoa {amt} kila wiki, biashara yangu itafungwa."),
    ],
    "Embezzlement": [
        ("{proj} has disappeared and {who} cannot account for it.",
         "{proj} zimepotea na {who} hawezi kueleza zilienda wapi.",
         "{proj} imepotea na {who} haezi explain ilienda wapi.",
         "{proj} imepotea na {who} hawezi sema ilienda wapi."),
        ("{t}{who} and the committee shared {proj} among themselves.",
         "{t}{who} na kamati waligawana {proj}.",
         "{t}{who} na committee walishare {proj} kati yao.",
         "{t}{who} na committee walikula {proj} wakagawana."),
        ("{t}{who} moved {proj} into {his} personal account.",
         "{t}{who} alihamisha {proj} kwenye akaunti yake binafsi.",
         "{t}{who} ali-transfer {proj} kwa account yake ya personal.",
         "{t}{who} alichota {proj} akaweka kwa account yake."),
        ("We were told {proj} was spent, but nothing has been done on the ground.",
         "Tuliambiwa {proj} zimetumika, lakini hakuna kilichofanyika.",
         "Tuliambiwa {proj} ilitumika lakini hakuna kitu tumeona on the ground.",
         "Tuliambiwa {proj} imetumika lakini hakuna kitu imefanyika mtaa."),
        ("{who} has built a big house since {he} started handling {proj}.",
         "{who} amejenga nyumba kubwa tangu aanze kusimamia {proj}.",
         "{who} amejenga mansion tangu aanze ku-handle {proj}.",
         "{who} amejenga mjengo kubwa tangu aanze kushika {proj}."),
        ("Only half of {proj} reached the people; {who} kept the rest.",
         "Nusu tu ya {proj} ndiyo iliwafikia wananchi; {who} alibaki na zingine.",
         "Ni half tu ya {proj} ilifika kwa wananchi, {who} akabaki na zingine.",
         "Ni nusu tu ya {proj} ilifika kwa wasee, {who} akakula zingine."),
        ("{who} has been paying ghost workers from {proj}.",
         "{who} amekuwa akiwalipa wafanyakazi hewa kutoka {proj}.",
         "{who} amekuwa aki-pay ghost workers from {proj}.",
         "{who} analipa ghost workers kutumia {proj}."),
    ],
    "Procurement Fraud": [
        ("The tender for {item} was given to {firm}.",
         "Zabuni ya {item} ilipewa {firm}.",
         "Tender ya {item} ilipewa {firm}.",
         "Tender ya {item} ilipewa {firm}."),
        ("The county paid {a} for {item} that cost about {b} in the shops.",
         "Kaunti ililipa {a} kwa {item}, ilhali bei yake sokoni ni karibu {b}.",
         "County ililipa {a} kwa {item} ambazo bei yake ni kama {b} kwa shops.",
         "County ililipa {a} kwa {item} ilhali bei ni kama {b} tu madukani."),
        ("The tender for {item} was awarded before the advert had even closed.",
         "Zabuni ya {item} ilitolewa kabla tangazo halijafungwa.",
         "Tender ya {item} ili-awardiwa hata kabla advert ifunge.",
         "Tender ya {item} ilikuwa imeshapeanwa kabla advert iishe."),
        ("Only one bidder was allowed to submit for the tender for {item}; the others were turned away.",
         "Mzabuni mmoja tu ndiye aliruhusiwa kuwasilisha zabuni ya {item}, wengine wakafukuzwa.",
         "Ni bidder mmoja tu aliruhusiwa ku-submit tender ya {item}, wengine wakafukuzwa.",
         "Ni msee mmoja tu alikubaliwa kuapply tender ya {item}, wengine walifukuzwa."),
        ("The county paid {a} for {item}, but nothing was ever delivered.",
         "Kaunti ililipa {a} kwa ajili ya {item}, lakini hakuna kilicholetwa.",
         "County ililipa {a} for {item} lakini hakuna kitu ili-deliveriwa.",
         "County ililipa {a} juu ya {item} lakini hakuna kitu ililetwa."),
        ("What was supplied as {item} is substandard, yet the county paid {a}.",
         "Kilichosambazwa kama {item} ni duni, lakini kaunti ililipa {a}.",
         "Kitu ili-supply-iwa kama {item} ni substandard, lakini county ililipa {a}.",
         "Vile walisupply kama {item} ni fake tupu, lakini county ililipa {a}."),
        ("The evaluation committee changed the scores so that {firm} would win the tender for {item}.",
         "Kamati ya tathmini ilibadilisha alama ili {firm} ishinde zabuni ya {item}.",
         "Evaluation committee ilibadilisha marks ndio {firm} ishinde tender ya {item}.",
         "Committee walichange marks ndio {firm} ikule tender ya {item}."),
    ],
    "Abuse of Office": [
        ("{who} uses the county vehicle to take {his} children to school every day.",
         "{who} hutumia gari la kaunti kuwapeleka watoto wake shuleni kila siku.",
         "{who} anatumia gari ya county kupeleka watoto wake school kila siku.",
         "{who} anatumia gari ya county kupeleka watoi wake shule kila siku."),
        ("{who} makes junior staff work on {his} private farm during office hours.",
         "{who} huwalazimisha wafanyakazi wadogo kufanya kazi shambani kwake wakati wa kazi.",
         "{who} anafanya junior staff wakulime shamba lake during office hours.",
         "{who} anafanya wafanyikazi wadogo wamlimie shamba saa za job."),
        ("{t}{who} used the police to harass a man {he} had a personal dispute with.",
         "{t}{who} alitumia polisi kumhangaisha mtu waliyekuwa na ugomvi wa kibinafsi.",
         "{t}{who} alitumia police ku-harass mtu walikuwa na beef ya personal.",
         "{t}{who} alitumia makarao kumhanda jamaa walikuwa na beef."),
        ("{who} fills {his} family's private cars using the office fuel card.",
         "{who} hujaza mafuta magari ya familia yake kwa kadi ya mafuta ya ofisi.",
         "{who} anajaza magari ya family yake kutumia fuel card ya office.",
         "{who} anajaza mamotii ya familia yake na fuel card ya ofisi."),
        ("{t}{who} sacked a worker who refused to campaign for {his} preferred candidate.",
         "{t}{who} alimfuta kazi mfanyakazi aliyekataa kumfanyia kampeni mgombea wake.",
         "{t}{who} ali-fire worker aliyekataa ku-campaign for candidate wake.",
         "{t}{who} alimfuta job msee alikataa kumpigia kampeni candidate wake."),
        ("{who} uses the county lorry to carry sand for {his} private building.",
         "{who} anatumia lori la kaunti kubeba mchanga wa ujenzi wake binafsi.",
         "{who} anatumia lorry ya county kubeba sand ya construction yake.",
         "{who} anatumia lorry ya county kubeba mchanga ya mjengo yake."),
        ("{who} keeps the county ambulance at {his} home and uses it for family errands.",
         "{who} huweka ambulensi ya kaunti nyumbani kwake na kuitumia kwa shughuli za familia.",
         "{who} anaweka ambulance ya county kwake na anaitumia kwa errands za family.",
         "{who} anapark ambulance ya county kwake na anaitumia kwa mambo za familia."),
        ("{t}{who} ordered officers to arrest traders who had criticised {him} at a public meeting.",
         "{t}{who} aliamuru maafisa wawakamate wafanyabiashara waliomkosoa kwenye baraza.",
         "{t}{who} aliorder officers wa-arrest traders walimkosoa kwa public meeting.",
         "{t}{who} aliamrisha makarao washike wafanyibiashara walimkosoa kwa baraza."),
    ],
    "Nepotism/Favoritism": [
        ("{opp} went to {who}'s {rel}, who did not even attend the interview.",
         "{opp} ilipewa {rel} wa {who} ambaye hata hakuhudhuria usaili.",
         "{opp} ilipewa {rel} wa {who} ambaye hata haku-attend interview.",
         "{opp} ilipewa {rel} wa {who} na hata hakuja interview."),
        ("{t}{who} gave {opp} to {his} {rel} and ignored better-qualified applicants.",
         "{t}{who} alimpa {rel} wake {opp} na kuwapuuza waombaji waliohitimu zaidi.",
         "{t}{who} alipea {rel} wake {opp} aka-ignore watu wenye qualifications.",
         "{t}{who} alimpea {rel} wake {opp} akaacha wasee wenye ma-qualifications."),
        ("Everyone who got {opp} this year is related to {who}, starting with {his} {rel}.",
         "Kila aliyepata {opp} mwaka huu ni ndugu wa {who}, kuanzia na {rel} wake.",
         "Kila mtu alipata {opp} mwaka huu ni relative wa {who}, starting na {rel} wake.",
         "Kila msee alipata {opp} mwaka huu ni wa familia ya {who}, kuanzia {rel} wake."),
        ("The shortlist for {opp} was changed so that {who}'s {rel} could be included.",
         "Orodha ya walioteuliwa kwa {opp} ilibadilishwa ili {rel} wa {who} aingizwe.",
         "Shortlist ya {opp} ilibadilishwa ndio {rel} wa {who} aingie.",
         "Shortlist ya {opp} ilibadilishwa ndio {rel} wa {who} apenye."),
        ("{who} only gives {opp} to people from {his} own community.",
         "{who} huwapa watu wa jamii yake tu {opp}.",
         "{who} anapea tu watu wa community yake {opp}.",
         "{who} anapea tu wasee wa kabila yake {opp}."),
        ("{who}'s {rel} got {opp} with lower grades than many of us who were turned away.",
         "{rel} wa {who} alipata {opp} akiwa na alama za chini kuliko wengi wetu tuliokataliwa.",
         "{rel} wa {who} alipata {opp} na grades ziko chini kuliko wengi wetu tulikataliwa.",
         "{rel} wa {who} alipata {opp} na grades za chini kushinda wengi wetu tulikatwa."),
    ],
    "Fraud/Forgery of Documents": [
        ("A clerk at {office} is selling fake copies of the {doc} for {amt}.",
         "Karani wa {office} anauza nakala bandia za {doc} kwa {amt}.",
         "Clerk wa {office} anauza fake copies za {doc} kwa {amt}.",
         "Clerk wa {office} anauza {doc} feki kwa {amt}."),
        ("{who} used a fake {doc} to get the job.",
         "{who} alitumia {doc} bandia kupata kazi.",
         "{who} alitumia {doc} fake kupata job.",
         "{who} alitumia {doc} feki kupata job."),
        ("{t}officers at the lands office forged a title deed and our family land was transferred to a stranger.",
         "{t}maafisa wa ofisi ya ardhi walighushi hati ya shamba na ardhi ya familia yetu ikahamishiwa mgeni.",
         "{t}officers wa lands walighushi title deed na shamba ya family yetu ikatransferiwa kwa stranger.",
         "{t}wasee wa lands walipika title feki na shamba ya familia ikahamishwa kwa msee hatujui."),
        ("{t}someone at {office} forged my signature on a {doc} application.",
         "{t}mtu wa {office} alighushi sahihi yangu kwenye ombi la {doc}.",
         "{t}mtu wa {office} alighushi signature yangu kwa application ya {doc}.",
         "{t}msee wa {office} alifake sign yangu kwa application ya {doc}."),
        ("A broker outside {office} prints a {doc} in one day for {amt}, with no checks at all.",
         "Dalali nje ya {office} huchapisha {doc} kwa siku moja kwa {amt} bila ukaguzi wowote.",
         "Broker nje ya {office} anaprint {doc} kwa siku moja kwa {amt}, no checks.",
         "Kuna broker nje ya {office} anaprintia wasee {doc} siku moja kwa {amt}, hakuna kuchekiwa."),
        ("The receipts for the county project were altered to show payments that were never made.",
         "Stakabadhi za mradi wa kaunti zilibadilishwa kuonyesha malipo ambayo hayakufanywa.",
         "Receipts za project ya county zili-alteriwa kuonyesha payments ambazo hazikufanyika.",
         "Risiti za project ya county zilichezewa zionyeshe malipo hazikufanyika."),
        ("{t}an officer at {office} gave a foreigner a fake {doc} in exchange for {amt}.",
         "{t}afisa wa {office} alimpa mgeni {doc} bandia kwa {amt}.",
         "{t}officer wa {office} alipea foreigner {doc} fake kwa {amt}.",
         "{t}msee wa {office} alipea msee wa nje {doc} feki kwa {amt}."),
    ],
    "Conflict of Interest": [
        ("{who}'s own company won the contract to supply {item}, and {he} never declared it.",
         "Kampuni ya {who} mwenyewe ndiyo ilishinda kandarasi ya kusambaza {item}, bila yeye kutangaza.",
         "Company ya {who} mwenyewe ndio ilishinda contract ya ku-supply {item}, na haku-declare.",
         "Kampuni ya {who} mwenyewe ndio ilikula contract ya kusupply {item}, hakusema."),
        ("{t}{who} awarded the tender for {item} to a firm owned by {his} {rel}.",
         "{t}{who} alitoa zabuni ya {item} kwa kampuni ya {rel} wake.",
         "{t}{who} alipea company ya {rel} wake tender ya {item}.",
         "{t}{who} alipea kampuni ya {rel} wake tender ya {item}."),
        ("{who} sat on the panel that picked {his} own company to supply {item}.",
         "{who} alikuwa kwenye jopo lililochagua kampuni yake kusambaza {item}.",
         "{who} alikuwa kwa panel iliyochagua company yake ku-supply {item}.",
         "{who} alikuwa kwa panel ilichagua kampuni yake isupply {item}."),
        ("{who} has shares in the company that supplies {item} to the county.",
         "{who} ana hisa katika kampuni inayosambaza {item} kwa kaunti.",
         "{who} ana shares kwa company ina-supply {item} kwa county.",
         "{who} ana shares kwa kampuni inasupply {item} kwa county."),
        ("{t}{who} approved a payment of {a} to a company where {his} {rel} is a director.",
         "{t}{who} aliidhinisha malipo ya {a} kwa kampuni ambayo {rel} wake ni mkurugenzi.",
         "{t}{who} ali-approve payment ya {a} kwa company ambapo {rel} wake ni director.",
         "{t}{who} aliapprove malipo ya {a} kwa kampuni {rel} wake ni director."),
        ("{who} rents out {his} own building to the county office for {a} a year.",
         "{who} amekodisha jengo lake kwa ofisi ya kaunti kwa {a} kwa mwaka.",
         "{who} ame-lease building yake kwa county office kwa {a} per year.",
         "{who} amekodisha jengo yake kwa county kwa {a} kila mwaka."),
    ],
    "Other": [
        ("{office} has been closed for two weeks and nobody is there to help.",
         "{office} imefungwa kwa wiki mbili na hakuna mtu wa kusaidia.",
         "{office} imefungwa for two weeks na hakuna mtu wa kusaidia.",
         "{office} imefungwa wiki mbili na hakuna msee wa kusaidia."),
        ("My file keeps getting lost at {office} and nobody takes responsibility.",
         "Faili yangu hupotea kila mara katika {office} na hakuna anayewajibika.",
         "File yangu inapotea kila mara pale {office} na hakuna mtu anachukua responsibility.",
         "File yangu inapotea kila saa pale {office} na hakuna mtu anaresponsible."),
        ("At {office} the queues are so long that I have gone three times without being served.",
         "Katika {office} foleni ni ndefu mno; nimeenda mara tatu bila kuhudumiwa.",
         "Pale {office} queue ni ndefu sana, nimeenda mara tatu bila ku-servewa.",
         "Pale {office} laini ni ndefu mbaya, nimeenda mara tatu bila kuhudumiwa."),
        ("The officers at {office} are rude and shout at old people.",
         "Maafisa katika {office} ni wakorofi na huwafokea wazee.",
         "Officers pale {office} ni rude na wanafokea wazee.",
         "Maofficer pale {office} ni wakali na wanabonga vibaya na wazee."),
        ("The system at {office} has been down for a week, so nobody can be served.",
         "Mfumo katika {office} umekuwa chini kwa wiki moja, hakuna anayehudumiwa.",
         "System pale {office} imekuwa down for a week, hakuna mtu ana-servewa.",
         "System pale {office} iko down wiki nzima, hakuna msee anahudumiwa."),
        ("Officers at {office} report to work at eleven and leave at three.",
         "Maafisa wa {office} huingia kazini saa tano na kuondoka saa tisa.",
         "Officers wa {office} wanakuja job saa tano na wanaenda saa tisa.",
         "Maofficer wa {office} wanaingia job saa tano na wanaishia saa tisa."),
        ("I applied at {office} three months ago and still have no response.",
         "Niliomba katika {office} miezi mitatu iliyopita na bado sijapata jibu.",
         "Ni-apply pale {office} three months ago na bado sija-get response.",
         "Niliapply pale {office} miezi tatu imepita na bado sijaskia kitu."),
        ("There is no water or toilet at {office}, and people wait outside in the sun all day.",
         "Hakuna maji wala choo katika {office}, na watu husubiri juani siku nzima.",
         "Hakuna maji ama toilet pale {office}, na watu wanangoja nje kwa jua siku nzima.",
         "Hakuna maji ama choo pale {office}, wasee wanangoja kwa jua siku nzima."),
    ],
}
SMALL_AMOUNTS = [200, 500, 1000, 2000, 3000, 5000, 10000, 20000, 50000]
MONEY_CATEGORIES = ("Bribery", "Extortion")
EXTORTION_ARREST = (0, 1, 6, 8, 9)      # traffic police, county askari, chief, police station
EXTORTION_BUSINESS = (1, 10, 17)        # county askari, public health officer, revenue officer
# Words that keep their capital letter after an opening such as "Yesterday, ".
KEEP_CAPITAL = {"I", "Huduma", "Nyayo", "Inua", "Kenya", "Form"} | set(TOWNS)


def _join(opening: str, sentence: str) -> str:
    if not opening:
        return sentence
    first = sentence.split(" ", 1)[0]
    if first not in KEEP_CAPITAL and not (len(first) > 1 and first[1].isupper()):
        sentence = sentence[0].lower() + sentence[1:]
    return opening + sentence


def scenario(category: str, rng: random.Random) -> dict:
    """Everything that is fixed for one incident, whatever the register."""
    s: dict = dict(category=category, town=rng.choice(TOWNS), template=rng.choice(T[category]),
                   pron=rng.choice(PRONOUNS), pay=rng.choice(PAYMENT))
    if category in MONEY_CATEGORIES:
        allowed = range(len(SETTINGS))
        if category == "Extortion":   # only officials who can arrest, or close a business, make these threats
            allowed = EXTORTION_BUSINESS if T[category].index(s["template"]) == 5 else EXTORTION_ARREST
        setting = SETTINGS[rng.choice(list(allowed))]
        s.update(off=setting["off"], place=setting["place"], svc=setting["svc"], inf=setting["inf"],
                 amt=rng.choice(SMALL_AMOUNTS))
    else:
        s.update(who=rng.choice(OFFICIALS))
    if category == "Embezzlement":
        s["proj"] = rng.choice(PROJECTS)
    if category in ("Procurement Fraud", "Conflict of Interest"):
        low = rng.choice([50000, 100000, 200000, 500000])
        s.update(item=rng.choice(ITEMS), firm=rng.choice(FIRMS), a=low * rng.choice([3, 4, 5]), b=low)
    if category in ("Nepotism/Favoritism", "Conflict of Interest"):
        s.update(rel=rng.choice(RELATIVES), opp=rng.choice(OPPORTUNITIES))
    if category == "Fraud/Forgery of Documents":
        which = T[category].index(s["template"])
        pool = JOB_DOCUMENTS if which == 1 else IDENTITY_DOCUMENTS if which == 6 else range(len(DOCUMENTS))
        doc, office = DOCUMENTS[rng.choice(list(pool))]
        s.update(doc=doc, office=office, amt=rng.choice([1000, 2000, 3000, 5000, 10000, 20000]))
    if category == "Other":
        s["office"] = rng.choice(OFFICES)
    # Opening: a time (past-tense incidents only), a place, or nothing. A place
    # opening is used only where the report does not already say where it happened.
    has_time = s["template"][0].startswith("{t}")
    names_place = category in MONEY_CATEGORIES or "{town}" in s.get("office", ("",))[0] or \
        "lands office" in s["template"][0]
    roll = rng.random()
    s["opening"] = (rng.choice(TIMES) if has_time and roll < 0.45 else
                    PLACES[0] if roll > 0.75 and not names_place else ("", "", "", ""))
    roll = rng.random()
    if category in MONEY_CATEGORIES and roll < 0.25:
        options = MONEY_CLOSINGS if s["pay"][0] != " in cash" else MONEY_CLOSINGS[1:]
        s["closing"] = rng.choice(options)
    elif roll < 0.5:
        s["closing"] = rng.choice(CLOSINGS)
    else:
        s["closing"] = ("", "", "", "")
    return s


def render(s: dict, register: int, rng: random.Random) -> str:
    town = s["town"]

    def pick(name):
        return s[name][register].replace("{town}", town) if name in s else ""

    text = s["template"][register].replace("{t}", "").format(
        off=pick("off"), place=pick("place"), svc=pick("svc"), inf=pick("inf"), pay=s["pay"][register],
        who=pick("who"), proj=pick("proj"), item=pick("item"), firm=pick("firm"), rel=pick("rel"), opp=pick("opp"),
        doc=pick("doc"), office=pick("office"),
        amt=money(s["amt"], register, rng) if "amt" in s else "",
        a=money(s["a"], register, rng) if "a" in s else "",
        b=money(s["b"], register, rng) if "b" in s else "",
        **(s["pron"] if register == 0 else dict(he="", him="", his="")),
    )
    text = " ".join(text.split()).replace(" ,", ",").replace(" .", ".")
    text = text[0].upper() + text[1:]
    text = _join(s["opening"][register].replace("{town}", town), text)
    text += s["closing"][register]
    if INTERJECTIONS[register] and rng.random() < (0.25 if register == 3 else 0.1):
        text = _join(rng.choice(INTERJECTIONS[register]), text)
    if register == 3 and rng.random() < 0.3:      # Sheng is often typed casually
        text = text.rstrip(".").lower()
    elif register == 2 and rng.random() < 0.15:
        text = text.rstrip(".")
    return text


def main() -> int:
    rng = random.Random(SEED)
    rows, seen = [], set()
    for n in range(SCENARIOS):
        category = CATEGORIES[n % len(CATEGORIES)]
        group = "register-" + hashlib.sha1(f"{SEED}-{n}".encode()).hexdigest()[:10]
        # Redraw a scenario until none of its four texts repeats an earlier one:
        # the training pipeline rejects exact duplicates.
        while True:
            s = scenario(category, rng)
            texts = [render(s, register, rng) for register in range(len(REGISTERS))]
            if len(set(texts)) == len(texts) and not seen.intersection(texts):
                break
        seen.update(texts)
        for register, language in enumerate(REGISTERS):
            rows.append({"text": texts[register], "label": category,
                         "group_id": group, "language": language, "county": s["town"]})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "group_id", "language", "county"])
        writer.writeheader()
        writer.writerows(rows)
    OUT_META.write_text(json.dumps({
        "status": "synthetic_only_not_for_real_world_validation_or_production_training",
        "rows": len(rows), "scenarios": SCENARIOS, "registers": list(REGISTERS), "seed": SEED,
        "unique_texts": len({row["text"] for row in rows}),
        "note": "Fictional, template-generated reports: each scenario written in English, Kiswahili, "
                "mixed English-Kiswahili and Sheng. Judge it by its effect on text it was not generated from.",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows ({SCENARIOS} scenarios x {len(REGISTERS)} registers) to {OUT_CSV.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
