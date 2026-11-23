"""Generate a larger, report-form-like synthetic corpus for ML pipeline testing.

Every row is fictional. The narratives mimic the fields collected by the Kenya
Watch submission form, such as an incident location, date, role, requested
action, amount, and available evidence. This is a harder and more varied
synthetic benchmark, not a collection of real reports.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from ml.src.dataset import CATEGORIES

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"
OUT_CSV = OUT_DIR / "kenya_watch_synthetic_realistic.csv"
OUT_METADATA = OUT_DIR / "kenya_watch_synthetic_realistic.metadata.json"
SEED = 20261122
INCIDENTS_PER_CATEGORY = 60

COUNTIES = (
    ("Nairobi", "Embakasi East"),
    ("Mombasa", "Likoni"),
    ("Kisumu", "Kisumu Central"),
    ("Nakuru", "Nakuru East"),
    ("Eldoret", "Kapseret"),
    ("Machakos", "Machakos Central"),
    ("Garissa", "Garissa Township"),
    ("Kilifi", "Kilifi North"),
    ("Bungoma", "Bungoma Central"),
    ("Meru", "Imenti Central"),
)

AMOUNTS = ("KSh 300", "KSh 500", "KSh 1,000", "KSh 2,500", "KSh 5,000", "KSh 20,000")
DATES = (
    "3 January 2026", "17 February 2026", "9 March 2026", "22 April 2026",
    "6 May 2026", "14 June 2026", "28 July 2026", "11 August 2026",
)


PATTERNS: dict[str, tuple[tuple[str, ...], ...]] = {
    "Bribery": (
        ("the lands clerk", "a land-title search", "KSh 500", "asked for money before opening the file",
         "karani wa ardhi", "upekuzi wa hati ya ardhi", "aliomba pesa kabla ya kufungua faili"),
        ("the revenue officer", "a business permit", "KSh 1,000", "demanded an unofficial fee to approve the permit",
         "afisa wa mapato", "leseni ya biashara", "alidai ada isiyo rasmi ili kuidhinisha leseni"),
        ("the public health inspector", "a food-premises certificate", "KSh 300", "asked for a gift before signing the certificate",
         "mkaguzi wa afya", "cheti cha eneo la chakula", "aliomba zawadi kabla ya kutia sahihi kwenye cheti"),
        ("the traffic officer", "release of a matatu", "KSh 2,500", "said the vehicle would be held unless payment was made",
         "afisa wa trafiki", "kurejeshwa kwa matatu", "alisema magari yangeshikwa kama pesa hazitolipwa"),
        ("the county askari", "permission to continue trading", "KSh 500", "offered to overlook the stall issue for a private payment",
         "askari wa kaunti", "ruhusa ya kuuza sokoni", "aliahidi kuacha tatizo la kibanda kwa malipo ya faragha"),
        ("the immigration officer", "release of a passport", "KSh 5,000", "requested money to place the application ahead of the queue",
         "afisa wa uhamiaji", "utoaji wa pasipoti", "aliomba pesa kuingiza ombi mbele ya foleni"),
        ("the Huduma Centre clerk", "printing an identification document", "KSh 300", "asked for a personal payment to speed up the ordinary service",
         "karani wa Huduma Centre", "uchapishaji wa kitambulisho", "aliomba malipo ya binafsi kuharakisha huduma ya kawaida"),
        ("the police abstract clerk", "an approved police abstract", "KSh 1,000", "said the document would not be issued without an extra amount",
         "karani wa polisi", "abstract ya polisi", "alisema hati haitaandikwa bila kiasi cha ziada"),
    ),
    "Embezzlement": (
        ("the CDF committee", "KSh 20,000", "borehole money", "was withdrawn but the borehole was never completed",
         "kamati ya CDF", "shilingi elfu ishirini", "fedha za kisima", "zilitozwa lakini kisima hakikujengwa"),
        ("the bursary account clerk", "KSh 5,000", "student bursary funds", "entered receipts that could not be matched to payments",
         "karani wa akaunti ya basari", "shilingi elfu tano", "fedha za wanafunzi", "aliingiza risiti zisizofanana na malipo"),
        ("the school feeding committee", "KSh 20,000", "lunch programme funds", "were used to buy items not delivered to the school",
         "kamati ya chakula cha shule", "shilingi elfu ishirini", "fedha za chakula", "zilitumika kununua bidhaa ambazo hazikupelekwa shuleni"),
        ("the county hospital accountant", "KSh 20,000", "medicine purchase money", "paid a supplier for drugs that are absent from the store",
         "mhasibu wa hospitali ya kaunti", "shilingi elfu ishirini", "pesa za dawa", "alilipa msambazaji wa dawa ambazo hazipo ghalani"),
        ("the public works foreman", "KSh 5,000", "road-repair funds", "were spent while the damaged section remained unfinished",
         "msimamizi wa kazi za umma", "shilingi elfu tano", "fedha za ukarabati wa barabara", "zilitumika huku sehemu iliyoharibika bado haijarekebishwa"),
        ("the youth fund administrator", "KSh 20,000", "approved youth project money", "was transferred to a personal account",
         "msimamizi wa mfuko wa vijana", "shilingi elfu ishirini", "pesa ya mradi wa vijana", "ilihamishiwa akaunti ya binafsi"),
        ("the county market cashier", "KSh 2,500", "weekly market fee collections", "were missing from the ledger and bank deposit",
         "mhasibu wa soko la kaunti", "shilingi elfu mbili na mia tano", "mapato ya ada za soko", "hayakuonekana kwenye daftari wala amana ya benki"),
        ("the relief distribution committee", "KSh 5,000", "food and blankets", "were recorded as delivered but never reached beneficiaries",
         "kamati ya usambazaji wa msaada", "shilingi elfu tano", "chakula na blanketi", "ziliandikwa kuwa zimewasilishwa lakini hazikufika kwa walengwa"),
    ),
    "Procurement Fraud": (
        ("the evaluation committee", "tender KW-026", "road-maintenance tender", "changed requirements after bids were opened",
         "kamati ya tathmini", "zabuni KW-026", "zabuni ya ukarabati wa barabara", "ilibadilisha masharti baada ya zabuni kufunguliwa"),
        ("the procurement officer", "tender HOS-118", "hospital equipment tender", "split one contract into smaller purchases to avoid open competition",
         "afisa wa ununuzi", "zabuni HOS-118", "zabuni ya vifaa vya hospitali", "aligawanya mkataba mmoja kuwa ununuzi mdogo ili kuepuka ushindani"),
        ("the tender panel", "tender SCH-204", "classroom construction tender", "awarded the bid to a company owned by a panel member",
         "jopo la zabuni", "zabuni SCH-204", "zabuni ya ujenzi wa madarasa", "ilitenga zabuni kwa kampuni inayomilikiwa na mjumbe wa jopo"),
        ("the county supply office", "contract SUP-77", "office furniture", "paid for 40 units but received only 12",
         "ofisi ya vifaa ya kaunti", "mkataba SUP-77", "samani za ofisi", "ililipa vitengo 40 lakini ikapokea 12"),
        ("the bid evaluation team", "tender ICT-09", "computer and network systems", "used scoring criteria that were not published in the notice",
         "timu ya tathmini ya zabuni", "zabuni ICT-09", "mifumo ya kompyuta na mtandao", "ilitumia vigezo vya alama ambavyo havikutangazwa"),
        ("the contract committee", "tender VEH-441", "vehicle leasing", "accepted a bid after the deadline while rejecting timely bidders",
         "kamati ya mikataba", "zabuni VEH-441", "ukodishaji wa magari", "ilikubali zabuni baada ya muda na kukataa zile zilizowasilishwa kwa wakati"),
        ("the procuring entity", "tender SEC-55", "security services", "gave one supplier confidential bid information",
         "taasisi ya ununuzi", "zabuni SEC-55", "huduma za ulinzi", "ilampa msambazaji mmoja taarifa za zabuni kwa siri"),
        ("the evaluation secretary", "contract FUEL-33", "fuel supply", "altered the comparison sheet after bids were submitted",
         "katibu wa tathmini", "mkataba FUEL-33", "usambazaji wa mafuta", "alibadilisha jedwali la kulinganisha baada ya zabuni kuwasilishwa"),
    ),
    "Abuse of Office": (
        ("the ward administrator", "blocked my valid application", "without recorded reasons", "because I challenged a previous decision",
         "msimamizi wa wadi", "alizuia ombi langu halali", "bila sababu iliyoandikwa", "kwa sababu nilipinga uamuzi wa awali"),
        ("the chief", "used public staff", "to collect campaign materials", "during official working hours",
         "chifu", "alitumia wafanyakazi wa umma", "kukusanya vifaa vya kampeni", "wakati wa saa za kazi"),
        ("the licensing manager", "cancelled a lawful business licence", "after a personal dispute", "and did not give the owner a review procedure",
         "meneja wa leseni", "alifuta leseni halali ya biashara", "baada ya mzozo binafsi", "bila kumpa mmiliki utaratibu wa kukata rufaa"),
        ("the public programme head", "reserved a vacant opportunity", "for a person outside the published process", "and ordered staff to alter the applicant list",
         "mkuu wa programu ya umma", "alihifadhi nafasi", "kwa mtu nje ya utaratibu wa wazi", "akaagiza wafanyakazi kubadilisha orodha"),
        ("the department director", "used an official vehicle", "for a private family trip", "and signed the fuel claim through the office",
         "mkurugenzi wa idara", "alitumia gari rasmi", "kwa safari ya familia binafsi", "akatia sahihi madai ya mafuta kupitia ofisi"),
        ("the records officer", "refused access to my own file", "despite an approved request", "and threatened to remove the request record",
         "afisa wa rekodi", "alikataa kunipa faili yangu", "hata baada ya ombi laidhinishwa", "akatishia kuondoa rekodi ya ombi"),
        ("the school principal", "directed a teacher", "to use school equipment for a private function", "during school time",
         "mkuu wa shule", "aliamuru mwalimu", "atumie vifaa vya shule kwa hafla binafsi", "wakati wa shule"),
        ("the county officer", "changed a queue order", "to serve a personal acquaintance first", "while others were told to wait",
         "afisa wa kaunti", "alibadilisha utaratibu wa foleni", "kumhudumia mtu anayemfahamu kwanza", "wengine wakaambiwa wangojee"),
    ),
    "Nepotism/Favoritism": (
        ("the county recruitment panel", "appointed the chairperson's cousin", "a training office job", "although the cousin had not applied",
         "jopo la ajira la kaunti", "alimteua binamu wa mwenyekiti", "katika ofisi ya mafunzo", "ingawa binamu hakuwa ameomba"),
        ("the school committee", "awarded a supply contract", "to a relative's company", "after other bidders were excluded",
         "kamati ya shule", "ilipewa mkataba wa vifaa", "kampuni ya ndugu", "baada ya wazabuni wengine kutengwa"),
        ("the bursary committee", "gave the bursary", "to the committee secretary's child", "while several eligible applicants received nothing",
         "kamati ya basari", "ilitenga ufadhili", "kwa mtoto wa katibu wa kamati", "huku waombaji wengine waliostahili wakinyimwa"),
        ("the market allocation team", "allocated premium stalls", "to officials' friends", "before the published selection date",
         "timu ya mgao wa soko", "iligawia vibanda bora", "kwa marafiki wa maafisa", "kabla ya tarehe ya uteuzi iliyotangazwa"),
        ("the appointment board", "promoted an officer's spouse", "to a senior post", "despite a lower interview score",
         "bodi ya uteuzi", "alimpandisha mwenzi wa afisa", "na nafasi ya juu", "licha ya kupata alama ndogo kwenye usaili"),
        ("the CDF board", "funded the chairman's project", "ahead of equally eligible projects", "without an open vote",
         "bodi ya CDF", "ilifadhili mradi wa mwenyekiti", "kabla ya miradi mingine yenye sifa sawa", "bila kura ya wazi"),
        ("the public internship programme", "selected a local official's child", "for all available placements", "even though the position was not advertised",
         "programu ya mafunzo ya umma", "ilichagua mtoto wa afisa wa eneo", "kwa nafasi zote", "ingawa nafasi haikutangazwa"),
        ("the procurement committee", "gave repeated contracts", "to the chairperson's close associate", "while qualified suppliers were not invited",
         "kamati ya ununuzi", "ilitenga mikataba mara kwa mara", "kwa mshirika wa karibu wa mwenyekiti", "huku wasambazaji wenye sifa hawakualikwa"),
    ),
    "Extortion": (
        ("the police officer", "held my identification card", "KSh 2,500", "threatening arrest unless I paid in cash",
         "afisa wa polisi", "alishika kitambulisho changu", "shilingi elfu mbili na mia tano", "akitishia kunikamata nisipoleta pesa taslimu"),
        ("the county askari", "took the goods", "KSh 1,000", "said they would be returned only after a payment",
         "askari wa kaunti", "alichukua bidhaa", "shilingi elfu moja", "akasema zingerudishwa baada ya malipo"),
        ("the roadblock officer", "stopped my vehicle", "KSh 500", "threatened to make a false inspection fault",
         "afisa wa kizuizi", "alizuia gari langu", "shilingi mia tano", "akitishia kurekodi dosari ya uongo"),
        ("the inspector", "threatened to close my kiosk", "KSh 3,000", "and demanded an immediate private payment",
         "mkaguzi", "alitishia kufunga kibanda", "shilingi elfu tatu", "akadai malipo ya faragha mara moja"),
        ("the public officer", "threatened to blacklist my supplier account", "KSh 5,000", "unless I paid from my personal funds",
         "afisa wa umma", "alitishia kuondoa akaunti ya msambazaji wangu", "shilingi elfu tano", "nisipolipa kutoka kwa pesa yangu binafsi"),
        ("the station officer", "refused to release my brother", "KSh 2,000", "while warning that the case file could be damaged",
         "afisa wa stesheni", "alikataa kumwachilia kaka yangu", "shilingi elfu mbili", "akionya kuwa faili ya kesi ingeweza kuharibiwa"),
        ("the gate security supervisor", "blocked my lorry", "KSh 1,500", "threatening to call a taxing inspection",
         "msimamizi wa usalama langoni", "alizuia lori langu", "shilingi elfu moja na mia tano", "akitishia kupiga ukaguzi wa kodi"),
        ("the licensing inspector", "demanded cash", "KSh 800", "and said he would report a compliance breach if I refused",
         "mkaguzi wa leseni", "alidai pesa taslimu", "shilingi mia nane", "akasema angemripoti kwa uvunjaji wa sheria nikikataa"),
    ),
    "Fraud/Forgery of Documents": (
        ("a completion certificate", "the public works file", "was signed", "by a person whose office stamp was copied",
         "cheti cha kukamilisha kazi", "faili ya kazi za umma", "kilitiwa sahihi", "na mtu ambaye muhuri wake ulikuwashwa"),
        ("a delivery invoice", "the procurement file", "listed goods", "that were never received",
         "ankara ya uwasilishaji", "faili ya ununuzi", "iliandika bidhaa", "ambazo hazikupokelewa"),
        ("a land transfer record", "the county registry", "was altered", "to show a transfer that had not been approved",
         "rekodi ya uhamisho wa ardhi", "daftari la kaunti", "ilibadilishwa", "kuonyesha uhamisho ambao haukuidhinishwa"),
        ("a meeting minute", "the committee record", "was edited", "to include a resolution that members never passed",
         "kumbukumbu ya mkutano", "rekodi ya kamati", "ilihaririwa", "kuweka azimio ambalo wajumbe hawalikubali"),
        ("an academic certificate", "the appointment file", "contained a changed name", "and a stamp that the issuing school says is not genuine",
         "cheti cha elimu", "faili ya uteuzi", "kilikuwa na jina lililobadilishwa", "na muhuri ambao shule inasema si halisi"),
        ("a business permit", "the revenue office", "showed a false address", "and a permit number not in the official register",
         "leseni ya biashara", "ofisi ya mapato", "ilonyesha anwani ya uongo", "na nambari ya kibali isiyopo kwenye rejista rasmi"),
        ("a payment voucher", "the hospital account", "carried a duplicate signature", "for a purchase that has no supplier evidence",
         "vocha ya malipo", "akaunti ya hospitali", "ilikuwa na sahihi iliyorudishwa", "kwa ununuzi usio na ushahidi wa msambazaji"),
        ("a police clearance letter", "the public application", "used a fabricated stamp", "to make an unapproved clearance appear valid",
         "barua ya kibali cha polisi", "ombi la umma", "ilitumia muhuri wa kughushi", "kuifanya idhini isiyoidhinishwa ionekane halali"),
    ),
    "Conflict of Interest": (
        ("the procurement chairperson", "a supplier bid", "owned shares", "in the bidding company but did not declare it",
         "mwenyekiti wa ununuzi", "zabuni ya msambazaji", "alikuwa na hisa", "katika kampuni iliyozabuni bila kutangaza"),
        ("the project director", "a construction contract", "also owned", "the company selected for the work",
         "mkurugenzi wa mradi", "mkataba wa ujenzi", "pia alikuwa mwenye", "kampuni iliyochaguliwa kwa kazi"),
        ("the committee secretary", "a bursary award", "was paid", "by an organization that received the award",
         "katibu wa kamati", "ufadhili wa basari", "alilipwa", "na shirika ambalo lilipokea ufadhili"),
        ("the licensing board member", "a permit application", "was a director", "of the applicant company",
         "mjumbe wa bodi ya leseni", "ombi la kibali", "alikuwa mkurugenzi", "wa kampuni iliyowasilisha ombi"),
        ("the county finance officer", "a lease decision", "personally owned", "the land being leased to the public entity",
         "afisa wa fedha wa kaunti", "uamuzi wa upangishaji", "alikuwa mmiliki binafsi", "wa ardhi inayopangiwa taasisi ya umma"),
        ("the school board member", "a tender for uniforms", "employed", "the bidder as a private consultant",
         "mjumbe wa bodi ya shule", "zabuni ya sare", "aliwahi kuajiri", "mzabuni kama mshauri wake binafsi"),
        ("the grant panel chair", "a community grant", "was a partner", "in the organisation applying for funds",
         "mwenyekiti wa jopo la ruzuku", "ruzuku ya jamii", "alikuwa mshirika", "katika shirika lililoomba fedha"),
        ("the public appointment officer", "a job assessment", "was the spouse", "of one shortlisted candidate",
         "afisa wa uteuzi wa umma", "tathmini ya kazi", "alikuwa mwenzi", "wa mgombea mmoja aliyeorodheshwa"),
    ),
    "Other": (
        ("the public office", "closed early", "with no notice", "and did not publish when services would resume",
         "ofisi ya umma", "ilifungwa mapema", "bila taarifa", "wala haikutangaza wakati huduma ingerejea"),
        ("the service desk", "sent me between offices", "for the same document", "and nobody took responsibility",
         "dirisha la huduma", "lilinipeleka ofisi tofauti", "kwa hati ileile", "bila mtu yeyote kuchukua jukumu"),
        ("the water department", "left the community pump", "out of service", "for several months without an explanation",
         "idara ya maji", "iliacha pampu ya jamii", "kutofanya kazi", "kwa miezi kadhaa bila maelezo"),
        ("the public counter", "charged more than the published price", "for a standard form", "and gave no official receipt",
         "dirisha la umma", "lilitoza zaidi ya bei iliyotangazwa", "kwa fomu ya kawaida", "bila risiti rasmi"),
        ("the roads office", "promised resurfacing", "for our street", "but no work began before the deadline",
         "ofisi ya barabara", "iliahidi kuweka lami", "kwenye barabara yetu", "lakini hakuna kazi iliyoanza kabla ya muda"),
        ("the hospital desk", "refused to explain", "why the queue was suspended", "and gave no complaint channel",
         "dirisha la hospitali", "lilikataa kueleza", "kwa nini foleni ilisitishwa", "bila kutoa njia ya kulalamika"),
        ("the county portal", "removed a public notice", "without a published reason", "after residents requested services",
         "tovuti ya kaunti", "iliondoa taarifa ya umma", "bila sababu iliyochapishwa", "baada ya wakazi kuomba huduma"),
        ("the public help line", "did not answer", "for repeated calls", "and provided no follow-up reference",
         "simu ya msaada ya umma", "haikujibu", "baada ya simu nyingi", "bila nambari ya kufuatilia"),
    ),
}


def _english_text(category: str, pattern: tuple[str, str, str, str, str, str, str],
                  county: str, subcounty: str, date: str, index: int) -> str:
    role, event, detail, aftermath = pattern[:4]
    amount = AMOUNTS[index % len(AMOUNTS)]
    evidence = (
        "I kept the receipt and the message thread."
        if index % 3 == 0 else
        "I have a screenshot and the names of two people who witnessed it."
        if index % 3 == 1 else
        "I recorded the time and wrote down the office number; no identity of a witness is included here."
    )
    if category == "Bribery":
        return (
            f"On {date}, I went to the public office in {subcounty}, {county}, for {event}. "
            f"The {role} {aftermath}; the demand was {amount}. {evidence}"
        )
    if category == "Embezzlement":
        return (
            f"I am reporting irregular use of {event} in {subcounty}, {county}, after {date}. "
            f"Records show that {role} handled {detail}, and the funds worth {amount} {aftermath}. {evidence}"
        )
    if category == "Procurement Fraud":
        return (
            f"For {event} in {subcounty}, {county}, {role} {aftermath}. "
            f"The file concerns {detail} and an estimated value of {amount}. {evidence}"
        )
    if category == "Abuse of Office":
        return (
            f"On {date} in {subcounty}, {county}, {role} {event} {detail} {aftermath}. "
            f"This was not a routine decision. {evidence}"
        )
    if category == "Nepotism/Favoritism":
        return (
            f"In {subcounty}, {county}, on {date}, {role} {event} {detail} {aftermath}. "
            f"The estimated benefit or contract value was {amount}. {evidence}"
        )
    if category == "Extortion":
        return (
            f"On {date} in {subcounty}, {county}, {role} {event} and demanded {detail}, about {amount}. "
            f"They {aftermath}. {evidence}"
        )
    if category == "Fraud/Forgery of Documents":
        return (
            f"I checked {role} relating to {event} in {subcounty}, {county}, on {date}. "
            f"The document {detail} {aftermath}. The apparent difference is worth reviewing. {evidence}"
        )
    if category == "Conflict of Interest":
        return (
            f"On {date}, {role} took part in a decision about {event} in {subcounty}, {county}. "
            f"The officer {detail} {aftermath}. The financial interest was not publicly disclosed. {evidence}"
        )
    return (
        f"On {date}, I dealt with {role} in {subcounty}, {county}. "
        f"The office {event} {detail} {aftermath}. I asked for a response but received none. {evidence}"
    )


def _kiswahili_text(category: str, pattern: tuple[str, str, str, str, str, str, str],
                    county: str, subcounty: str, date: str, index: int) -> str:
    role = pattern[4]
    event = pattern[6] if len(pattern) == 8 else pattern[5]
    aftermath = pattern[-1]
    amount = AMOUNTS[index % len(AMOUNTS)].replace("KSh ", "Sh").replace(",", "")
    evidence = (
        "Nimehifadhi risiti na ujumbe wa mazungumzo."
        if index % 3 == 0 else
        "Nina picha ya skrini na majina ya mashahidi wawili."
        if index % 3 == 1 else
        "Nimeandikia saa na nambari ya ofisi bila kujumuisha utambulisho wa shahidi."
    )
    if category == "Bribery":
        return (
            f"Tarehe {date}, nilienda ofisi ya umma katika {subcounty}, {county}, kwa {event}. "
            f"{role} {aftermath}; {amount}. {evidence}"
        )
    if category == "Embezzlement":
        return (
            f"Ninawasilisha tuhuma za matumizi mabaya ya {event} katika {subcounty}, {county}, "
            f"tangu {date}. {role} {aftermath}. Nimeona rekodi zinazoashiria kiasi cha {amount}. {evidence}"
        )
    if category == "Procurement Fraud":
        return (
            f"Kwa {event} katika {subcounty}, {county}, {role} {aftermath}. "
            f"Tukio linahusu {event} yenye thamani ya {amount}. {evidence}"
        )
    if category == "Abuse of Office":
        return (
            f"Tarehe {date} katika {subcounty}, {county}, {role} {aftermath}. "
            f"Huu si uamuzi wa kawaida wa kazi. {evidence}"
        )
    if category == "Nepotism/Favoritism":
        return (
            f"Katika {subcounty}, {county}, tarehe {date}, {role} {aftermath}. "
            f"Thamani ya manufaa au mkataba ilikuwa {amount}. {evidence}"
        )
    if category == "Extortion":
        return (
            f"Tarehe {date} katika {subcounty}, {county}, {role} {aftermath}, {amount}. {evidence}"
        )
    if category == "Fraud/Forgery of Documents":
        return (
            f"Nimekagua hati kuhusu {event} katika {subcounty}, {county}, tarehe {date}. "
            f"{role} {aftermath}. {evidence}"
        )
    if category == "Conflict of Interest":
        return (
            f"Tarehe {date}, {role} alishiriki uamuzi kuhusu {event} katika {subcounty}, {county}. "
            f"Maslahi ya kifedha hayakutangazwa hadharani. {evidence}"
        )
    return (
        f"Tarehe {date}, nilishughulika na {role} katika {subcounty}, {county}. "
        f"{event} {aftermath}. {evidence}"
    )


def _mixed_text(category: str, pattern: tuple[str, str, str, str, str, str, str],
                county: str, subcounty: str, date: str, index: int) -> str:
    sw_role = pattern[4]
    sw_event = pattern[6] if len(pattern) == 8 else pattern[5]
    sw_aftermath = pattern[-1]
    amount = AMOUNTS[index % len(AMOUNTS)].replace("KSh ", "")
    evidence = "na screenshot na receipt bado" if index % 2 == 0 else "nimeweka details ya time na office number"
    if category == "Bribery":
        return f"Nilienda pale office ya {subcounty}, {county}, on {date} for {sw_event}. {sw_role} {sw_aftermath}, {amount}, {evidence}."
    if category == "Embezzlement":
        return f"Kuna money misuse kwa {sw_event} pale {subcounty}, {county}, after {date}. {sw_role} {sw_aftermath}, {amount}, {evidence}."
    if category == "Procurement Fraud":
        return f"Kwa tender ya {sw_event} pale {subcounty}, {county}, {sw_role} {sw_aftermath}. Value ilikuwa around {amount}, {evidence}."
    if category == "Abuse of Office":
        return f"On {date} pale {subcounty}, {county}, {sw_role} {sw_aftermath}; hii ilikuwa abuse ya ofisi, {evidence}."
    if category == "Nepotism/Favoritism":
        return f"Pale {subcounty}, {county}, on {date}, {sw_role} {sw_aftermath}, benefit ilikuwa {amount}, {evidence}."
    if category == "Extortion":
        return f"On {date} pale {subcounty}, {county}, {sw_role} {sw_aftermath}, walidai {amount}, {evidence}."
    if category == "Fraud/Forgery of Documents":
        return f"Nilikagua document for {sw_event} pale {subcounty}, {county}, on {date}; {sw_role} {sw_aftermath}, {evidence}."
    if category == "Conflict of Interest":
        return f"On {date} pale {subcounty}, {county}, {sw_role} joined the decision on {sw_event}; vested interest haikudeclear, {evidence}."
    return f"Pale {subcounty}, {county} on {date}, {sw_role} {sw_aftermath}; no action taken, {evidence}."


def _sheng_text(category: str, pattern: tuple[str, str, str, str, str, str, str],
                county: str, subcounty: str, date: str, index: int) -> str:
    detail = pattern[2]
    sw_role = pattern[4]
    sw_event = pattern[6] if len(pattern) == 8 else pattern[5]
    sw_aftermath = pattern[-1]
    amount = AMOUNTS[index % len(AMOUNTS)].replace("KSh ", "").replace(",", "")
    evidence = "nimehifadhi screenshot na receipt" if index % 2 == 0 else "nimeandika time, office n marafiki wawili waliona"
    if category == "Bribery":
        return f"Juzi {date} nikaingia pale office ya {subcounty}, {county}, for {sw_event}. {sw_role} {sw_aftermath} {amount}, {evidence}."
    if category == "Embezzlement":
        return f"Kuna doo za {sw_event} zilipotea pale {subcounty}, {county} after {date}. {sw_role} {sw_aftermath}, {amount}, {evidence}."
    if category == "Procurement Fraud":
        return f"Tender ya {sw_event} pale {subcounty}, {county} ilikuwa imepangwa. {sw_role} {sw_aftermath}, around {amount}, {evidence}."
    if category == "Abuse of Office":
        return f"On {date} pale {subcounty}, {county}, {sw_role} {sw_aftermath}; alitumia cheo vibaya, {evidence}."
    if category == "Nepotism/Favoritism":
        return f"Pale {subcounty}, {county} on {date}, {sw_role} {sw_aftermath}; benefit/value ilikuwa {amount}, {evidence}."
    if category == "Extortion":
        return f"On {date} pale {subcounty}, {county}, {sw_role} {sw_aftermath}, {amount}; kuni-threaten kama sikuli ipa, {evidence}."
    if category == "Fraud/Forgery of Documents":
        return f"Nili-check document ya {sw_event} pale {subcounty}, {county}, on {date}; {sw_role} {sw_aftermath}, {evidence}."
    if category == "Conflict of Interest":
        return f"On {date} pale {subcounty}, {county}, {sw_role} aliingia decision ya {sw_event}; {detail}, lakini haikudeclare, {evidence}."
    return f"Pale {subcounty}, {county} on {date}, {sw_role} {sw_aftermath}; no answer kuandaliwa, {evidence}."


def rows_for_dataset() -> list[dict[str, str]]:
    if set(PATTERNS) != set(CATEGORIES):
        raise ValueError("Realistic synthetic patterns must match the canonical category taxonomy.")

    rng = random.Random(SEED)
    rows: list[dict[str, str]] = []
    for category in CATEGORIES:
        for index in range(INCIDENTS_PER_CATEGORY):
            pattern = PATTERNS[category][index % len(PATTERNS[category])]
            county, subcounty = COUNTIES[(index * 3 + CATEGORIES.index(category)) % len(COUNTIES)]
            date = DATES[(index * 5 + CATEGORIES.index(category)) % len(DATES)]
            group_id = f"realistic-{category.lower().replace('/', '-').replace(' ', '-')}-{index + 1:02d}"
            variants = (
                ("English", _english_text(category, pattern, county, subcounty, date, index)),
                ("Kiswahili", _kiswahili_text(category, pattern, county, subcounty, date, index)),
                ("Mixed", _mixed_text(category, pattern, county, subcounty, date, index)),
                ("Sheng", _sheng_text(category, pattern, county, subcounty, date, index)),
            )
            for language, text in variants:
                if language == "English":
                    text = f"{text} Case note KW-{index + 1:03d}."
                elif language == "Kiswahili":
                    text = f"{text} Rejea KW-{index + 1:03d}."
                elif language == "Mixed":
                    text = f"{text} Case/rekodi KW-{index + 1:03d}."
                else:
                    text = f"{text} Ref KW-{index + 1:03d}."
                rows.append({
                    "text": text,
                    "label": category,
                    "group_id": group_id,
                    "language": language,
                    "county": county,
                })
    rng.shuffle(rows)
    return rows


def main() -> int:
    rows = rows_for_dataset()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("text", "label", "group_id", "language", "county"))
        writer.writeheader()
        writer.writerows(rows)

    languages: dict[str, int] = {}
    for row in rows:
        languages[row["language"]] = languages.get(row["language"], 0) + 1

    OUT_METADATA.write_text(json.dumps({
        "dataset_name": "Kenya Watch realistic report-form synthetic corpus",
        "status": "synthetic_only_not_for_real_world_validation_or_production_training",
        "generation_method": "Deterministic fictional narratives combining report-form fields, county/subcounty values, dates, roles, amounts and evidence cues",
        "rows": len(rows),
        "scenario_groups": len({row["group_id"] for row in rows}),
        "incidents_per_category": INCIDENTS_PER_CATEGORY,
        "language_rows": languages,
        "county_values_for_audit_only": [county for county, _ in COUNTIES],
        "limitations": [
            "Every narrative is fictional and is not an observation of a real Kenyan report.",
            "The mixed-language and Sheng versions were authored for pipeline testing and have not been reviewed by native-language reviewers.",
            "The narratives imitate report-form structure but do not establish real-world prevalence or class balance.",
            "Variants within a group describe one fictional incident and must stay in the same validation fold.",
            "Scores on this corpus are not a real-world accuracy estimate.",
        ],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "rows": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "by_language": languages,
        "csv": str(OUT_CSV),
        "warning": "Synthetic-only. Not evidence of real-world accuracy.",
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
