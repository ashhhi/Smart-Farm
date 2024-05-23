import matplotlib.pyplot as plt

# raw data
EfficientUnet1_50 = {'loss': [0.9676605463027954, 0.813288152217865, 0.7402551174163818, 0.6650182604789734, 0.6202786564826965, 0.5742417573928833, 0.5491214990615845, 0.5251398682594299, 0.5174431800842285, 0.49903616309165955, 0.48575031757354736, 0.46748924255371094, 0.46549656987190247, 0.4599578380584717, 0.4537174701690674, 0.4379732310771942, 0.4345480501651764, 0.43504324555397034, 0.4181172251701355, 0.4157683253288269, 0.4121471047401428, 0.3974420428276062, 0.4019653797149658, 0.3924887180328369, 0.3834032416343689, 0.384331077337265, 0.38274773955345154, 0.3768889307975769, 0.36770763993263245, 0.3630301356315613, 0.37173259258270264, 0.36405128240585327, 0.3604150712490082, 0.35210156440734863, 0.3542012870311737, 0.35307183861732483, 0.348288357257843, 0.3359820544719696, 0.3519803285598755, 0.3409794867038727, 0.33418363332748413, 0.3341786563396454, 0.3342615067958832, 0.330331414937973, 0.3286139667034149, 0.332731157541275, 0.3221970498561859, 0.32215410470962524, 0.3144471049308777, 0.31178948283195496], 'accuracy': [0.6247924566268921, 0.7157573699951172, 0.731645941734314, 0.7506972551345825, 0.7582381367683411, 0.7719006538391113, 0.7744971513748169, 0.7819030284881592, 0.779347836971283, 0.7837143540382385, 0.7883914113044739, 0.7977576851844788, 0.7938394546508789, 0.7937753200531006, 0.7966911196708679, 0.8033605813980103, 0.8035478591918945, 0.802922248840332, 0.8111719489097595, 0.8114529252052307, 0.8133544325828552, 0.8207375407218933, 0.8177297115325928, 0.8198429346084595, 0.825014054775238, 0.8244654536247253, 0.823689877986908, 0.8259255886077881, 0.8307169675827026, 0.8317661881446838, 0.8268290162086487, 0.8318644165992737, 0.8334679007530212, 0.8380846977233887, 0.834257960319519, 0.8395137786865234, 0.8412263989448547, 0.8454411029815674, 0.8360576033592224, 0.8414060473442078, 0.8461800217628479, 0.8474912643432617, 0.8456808924674988, 0.8429111242294312, 0.8473743796348572, 0.8428564667701721, 0.8475473523139954, 0.8486484289169312, 0.8542901873588562, 0.8551719188690186]}
EfficientUnet51_100 = {'loss': [0.322700560092926, 0.3200955390930176, 0.3135759234428406, 0.31050440669059753, 0.33311301469802856, 0.3088075518608093, 0.2983191907405853, 0.29356101155281067, 0.3017069697380066, 0.2980872094631195, 0.3187849223613739, 0.30082911252975464, 0.2882471978664398, 0.28559553623199463, 0.293775737285614, 0.2794458866119385, 0.29489824175834656, 0.2854157090187073, 0.2908296287059784, 0.27907347679138184, 0.2839742600917816, 0.2863032817840576, 0.2834804952144623, 0.27805039286613464, 0.2761746942996979, 0.273830384016037, 0.29419776797294617, 0.27376389503479004, 0.27261850237846375, 0.26384562253952026, 0.26167798042297363, 0.25536003708839417, 0.26895570755004883, 0.26295337080955505, 0.28250256180763245, 0.31033435463905334, 0.2703060805797577, 0.26803722977638245, 0.25548821687698364, 0.25590935349464417, 0.27188870310783386, 0.24983951449394226, 0.2739899754524231, 0.2670510411262512, 0.2731546461582184, 0.27178674936294556, 0.24341727793216705, 0.24104903638362885, 0.26458001136779785, 0.2610992193222046], 'accuracy': [0.8483363389968872, 0.8510078191757202, 0.8523892164230347, 0.8557039499282837, 0.8450684547424316, 0.8552597761154175, 0.8599115610122681, 0.862293004989624, 0.8589234352111816, 0.8630106449127197, 0.848703920841217, 0.8596080541610718, 0.8626753687858582, 0.8694218397140503, 0.8603208661079407, 0.8666054010391235, 0.8592355847358704, 0.8634538054466248, 0.863737940788269, 0.8668090105056763, 0.8652418851852417, 0.8638386726379395, 0.8643977046012878, 0.8685416579246521, 0.869247555732727, 0.8724456429481506, 0.8636513948440552, 0.8688597083091736, 0.8715884685516357, 0.8771700263023376, 0.8732317090034485, 0.877796471118927, 0.8698368668556213, 0.8773258328437805, 0.8674913048744202, 0.8564799427986145, 0.871467113494873, 0.8729802966117859, 0.8773513436317444, 0.8758779764175415, 0.8680903911590576, 0.8799116015434265, 0.8703299164772034, 0.8720951676368713, 0.8695619106292725, 0.8705227971076965, 0.8848429322242737, 0.8823406100273132, 0.875893771648407, 0.8774751424789429]}
EfficientUnet101_150 = {'loss': [0.2556157112121582, 0.25045228004455566, 0.24353833496570587, 0.24525336921215057, 0.2510729730129242, 0.24460913240909576, 0.27566245198249817, 0.24858735501766205, 0.24252672493457794, 0.23601330816745758, 0.24854706227779388, 0.24153317511081696, 0.24645152688026428, 0.24904660880565643, 0.2369789481163025, 0.23776064813137054, 0.24956347048282623, 0.23913350701332092, 0.23433326184749603, 0.24059903621673584, 0.22744104266166687, 0.22904300689697266, 0.24133533239364624, 0.24386420845985413, 0.23141711950302124, 0.24350400269031525, 0.22867092490196228, 0.22670868039131165, 0.23427855968475342, 0.22708040475845337, 0.2246955782175064, 0.2441990077495575, 0.2274036854505539, 0.243686243891716, 0.22805249691009521, 0.21709929406642914, 0.2254233956336975, 0.22168037295341492, 0.21833153069019318, 0.22758114337921143, 0.23331747949123383, 0.222205251455307, 0.2214137762784958, 0.21948851644992828, 0.2137458473443985, 0.2115534543991089, 0.21639086306095123, 0.2248438149690628, 0.2187507599592209, 0.2445211559534073], 'accuracy': [0.8792524933815002, 0.8813110589981079, 0.8795825839042664, 0.8810265064239502, 0.8802372813224792, 0.8832581639289856, 0.8656147718429565, 0.8783494234085083, 0.8816923499107361, 0.8853313326835632, 0.8823918700218201, 0.8845560550689697, 0.8797035217285156, 0.8808768391609192, 0.8810272216796875, 0.8830530047416687, 0.8793481588363647, 0.8821920156478882, 0.8883287906646729, 0.8823047280311584, 0.8873285055160522, 0.8869885206222534, 0.8850173950195312, 0.884789228439331, 0.8839892745018005, 0.8832216858863831, 0.8884636163711548, 0.8905444741249084, 0.883482038974762, 0.8844774961471558, 0.8867358565330505, 0.8782140016555786, 0.8867660760879517, 0.881065845489502, 0.8873268365859985, 0.8935093283653259, 0.8902643322944641, 0.8889589905738831, 0.8936192989349365, 0.8863968849182129, 0.8820118308067322, 0.8844000697135925, 0.8942919373512268, 0.888828694820404, 0.8916190266609192, 0.8957728743553162, 0.8905539512634277, 0.8898842930793762, 0.8928378820419312, 0.882378339767456]}
EfficientUnet151_200 = {'loss': [0.22148270905017853, 0.21768523752689362, 0.21787472069263458, 0.2071581929922104, 0.2055661827325821, 0.2079342156648636, 0.20707206428050995, 0.22116804122924805, 0.21518798172473907, 0.23637685179710388, 0.22788000106811523, 0.21946567296981812, 0.20465317368507385, 0.20307457447052002, 0.20087793469429016, 0.22899512946605682, 0.22686265408992767, 0.21527691185474396, 0.20368540287017822, 0.20702625811100006, 0.20528212189674377, 0.19668219983577728, 0.20109336078166962, 0.2008194774389267, 0.20353259146213531, 0.20840322971343994, 0.2076478898525238, 0.19957199692726135, 0.19794835150241852, 0.21979686617851257, 0.2025921642780304, 0.1968812346458435, 0.19685156643390656, 0.18974532186985016, 0.20388783514499664, 0.20207424461841583, 0.22500784695148468, 0.20979976654052734, 0.19832345843315125, 0.19587606191635132, 0.2024250477552414, 0.19897817075252533, 0.1875070333480835, 0.19665147364139557, 0.19611963629722595, 0.19249096512794495, 0.18387466669082642, 0.18164294958114624, 0.19644959270954132, 0.194617360830307], 'accuracy': [0.8890135884284973, 0.889044463634491, 0.892613410949707, 0.8968337774276733, 0.897468090057373, 0.8960415720939636, 0.899246335029602, 0.8898571133613586, 0.8959974646568298, 0.8845915198326111, 0.8858196139335632, 0.8889499306678772, 0.8984060883522034, 0.9014574885368347, 0.9005082845687866, 0.8896676898002625, 0.8876055479049683, 0.8939884305000305, 0.8975807428359985, 0.9024766087532043, 0.9002976417541504, 0.9047201871871948, 0.902753472328186, 0.9034144282341003, 0.9028732180595398, 0.9001467227935791, 0.8989930152893066, 0.9054414629936218, 0.9023952484130859, 0.8975135684013367, 0.9005329608917236, 0.9035683870315552, 0.9047845602035522, 0.9128226637840271, 0.9036539793014526, 0.8994436860084534, 0.898358941078186, 0.9012447595596313, 0.9045898914337158, 0.9097002744674683, 0.9071125388145447, 0.9117766618728638, 0.9139642119407654, 0.9075971841812134, 0.908520519733429, 0.9107137322425842, 0.9159202575683594, 0.9133070111274719, 0.9062247276306152, 0.9077133536338806]}

EfficientUnet3_5_1_50 = {'loss': [0.6147134900093079, 0.5351170897483826, 0.5048673152923584, 0.49009642004966736, 0.47141581773757935, 0.4715223014354706, 0.4521130323410034, 0.44074586033821106, 0.42701029777526855, 0.4164513647556305, 0.41293707489967346, 0.4151172339916229, 0.39172646403312683, 0.39528006315231323, 0.38529929518699646, 0.3752744197845459, 0.3756261467933655, 0.35119757056236267, 0.34981727600097656, 0.3477061986923218, 0.32194456458091736, 0.33247849345207214, 0.3202667832374573, 0.31624868512153625, 0.31874650716781616, 0.31648531556129456, 0.29262006282806396, 0.3045266270637512, 0.3022404611110687, 0.30048638582229614, 0.2908743917942047, 0.28610363602638245, 0.2853364050388336, 0.27313750982284546, 0.27273958921432495, 0.274771511554718, 0.26624658703804016, 0.2765866219997406, 0.25082215666770935, 0.2670253813266754, 0.26909273862838745, 0.2628011107444763, 0.254256933927536, 0.2638537883758545, 0.25301244854927063, 0.23972803354263306, 0.23462705314159393, 0.22928175330162048, 0.25349879264831543, 0.23406432569026947], 'accuracy': [0.7270828485488892, 0.7489092350006104, 0.7610595226287842, 0.770020604133606, 0.7814114093780518, 0.7785636782646179, 0.7901601195335388, 0.7932080626487732, 0.7973161935806274, 0.8062072992324829, 0.8083834052085876, 0.8032925724983215, 0.8183422684669495, 0.8155134916305542, 0.8167000412940979, 0.8213465809822083, 0.8218103051185608, 0.8375678062438965, 0.8350527286529541, 0.8313992619514465, 0.8439966440200806, 0.8417378664016724, 0.8421974182128906, 0.8467078804969788, 0.8509119153022766, 0.8468407988548279, 0.8571357131004333, 0.8533805012702942, 0.8538129925727844, 0.8529548645019531, 0.8561238646507263, 0.8625856041908264, 0.8574190735816956, 0.8680384755134583, 0.8669365644454956, 0.8649277687072754, 0.8676813244819641, 0.8669601678848267, 0.874550998210907, 0.8693303465843201, 0.8716347813606262, 0.8734860420227051, 0.8726230263710022, 0.8729979991912842, 0.8710128664970398, 0.8762020468711853, 0.8798086047172546, 0.8880542516708374, 0.8754132986068726, 0.8836148381233215]}
EfficientUnet3_5_51_100 = {'loss': [0.23416189849376678, 0.22654791176319122, 0.24073609709739685, 0.23485136032104492, 0.2288513332605362, 0.2188350260257721, 0.22002506256103516, 0.21091213822364807, 0.21758581697940826, 0.2324921190738678, 0.2299150973558426, 0.2305804342031479, 0.22502698004245758, 0.22279946506023407, 0.21602462232112885, 0.22726139426231384, 0.2377236783504486, 0.2160727083683014, 0.2018953263759613, 0.25590312480926514, 0.2069096565246582, 0.2060973346233368, 0.21230630576610565, 0.21177521347999573, 0.19657151401042938, 0.19752474129199982, 0.20041152834892273, 0.19519810378551483, 0.20022650063037872, 0.20644167065620422, 0.19395792484283447, 0.22065024077892303, 0.20083320140838623, 0.19959481060504913, 0.22063849866390228, 0.19973017275333405, 0.1970651000738144, 0.20894378423690796, 0.18972405791282654, 0.22489416599273682, 0.21487733721733093, 0.1900261789560318, 0.18687944114208221, 0.18295611441135406, 0.17980842292308807, 0.1922769993543625, 0.2237367182970047, 0.19974979758262634, 0.18798063695430756, 0.19623471796512604], 'accuracy': [0.8793500065803528, 0.8885403275489807, 0.883723795413971, 0.8809733986854553, 0.8865130543708801, 0.8850881457328796, 0.8874682784080505, 0.8903493285179138, 0.8874896764755249, 0.8802691102027893, 0.8830057382583618, 0.883108377456665, 0.8837423324584961, 0.8857418894767761, 0.8900417685508728, 0.8856650590896606, 0.8786078095436096, 0.8874514698982239, 0.8930909037590027, 0.8736437559127808, 0.893232524394989, 0.8896804451942444, 0.8929062485694885, 0.8955742120742798, 0.8936904072761536, 0.8989736437797546, 0.8926191926002502, 0.8966384530067444, 0.8918105959892273, 0.8909019231796265, 0.896647036075592, 0.8881471753120422, 0.8958523869514465, 0.897343635559082, 0.8877128958702087, 0.8958333730697632, 0.8976471424102783, 0.8924646973609924, 0.8993650078773499, 0.888081967830658, 0.8916594982147217, 0.8982493281364441, 0.9029585123062134, 0.9003649353981018, 0.9004170894622803, 0.9016197323799133, 0.8898407220840454, 0.8979840874671936, 0.8972839117050171, 0.8988614082336426]}
EfficientUnet3_5_101_150 = {'loss': [0.19806990027427673, 0.19046899676322937, 0.18740904331207275, 0.17671166360378265, 0.2177249938249588, 0.18529212474822998, 0.18939411640167236, 0.19825010001659393, 0.18960566818714142, 0.18733720481395721, 0.19202138483524323, 0.18438854813575745, 0.1930919736623764, 0.17784881591796875, 0.17269548773765564, 0.17440524697303772, 0.1765025556087494, 0.19364453852176666, 0.1841432899236679, 0.17631705105304718, 0.18148407340049744, 0.1762249618768692, 0.17975853383541107, 0.17807765305042267, 0.19884033501148224, 0.17262385785579681, 0.1684626191854477, 0.1730624884366989, 0.16918709874153137, 0.17527613043785095, 0.18497243523597717, 0.18585437536239624, 0.18359601497650146, 0.1817799061536789, 0.16425821185112, 0.1720532476902008, 0.17250558733940125, 0.1701980084180832, 0.178171306848526, 0.16753457486629486, 0.17806075513362885, 0.1673106551170349, 0.16333796083927155, 0.18024058640003204, 0.23009146749973297, 0.18092727661132812, 0.19052091240882874, 0.16790671646595, 0.16105014085769653, 0.15879100561141968], 'accuracy': [0.8937996625900269, 0.8990766406059265, 0.9002390503883362, 0.902115523815155, 0.8925085067749023, 0.9035660624504089, 0.9037798643112183, 0.895734429359436, 0.8975062966346741, 0.8993761539459229, 0.9013521671295166, 0.8990533947944641, 0.8951607346534729, 0.9048288464546204, 0.9040312170982361, 0.9034602642059326, 0.9021719694137573, 0.8976502418518066, 0.899695098400116, 0.9048250317573547, 0.9069136381149292, 0.9027884602546692, 0.9013282656669617, 0.9108109474182129, 0.8946338891983032, 0.9064205288887024, 0.9087411165237427, 0.9035014510154724, 0.9092972874641418, 0.9017699360847473, 0.8975440859794617, 0.9031064510345459, 0.9026712775230408, 0.9019627571105957, 0.9122447371482849, 0.903330385684967, 0.90663743019104, 0.9058887958526611, 0.9012604355812073, 0.9074789881706238, 0.9055710434913635, 0.9075266122817993, 0.9057139754295349, 0.9072650671005249, 0.884652853012085, 0.9035726189613342, 0.9024839997291565, 0.9067531228065491, 0.9124599695205688, 0.9101008772850037]}
EfficientUnet3_5_151_200 = {'loss': [0.16496634483337402, 0.1771135777235031, 0.17635048925876617, 0.1669171005487442, 0.19782406091690063, 0.17694734036922455, 0.16140280663967133, 0.1574167013168335, 0.16187605261802673, 0.17588140070438385, 0.17336136102676392, 0.1593170315027237, 0.16838183999061584, 0.20518140494823456, 0.17989470064640045, 0.17133067548274994, 0.15883290767669678, 0.17860805988311768, 0.16089415550231934, 0.1601637750864029, 0.16205987334251404, 0.15647903084754944, 0.15538039803504944, 0.15385015308856964, 0.20096486806869507, 0.20094501972198486, 0.1648423820734024, 0.15844199061393738, 0.1617206633090973, 0.15858836472034454, 0.15691165626049042, 0.17128658294677734, 0.17718398571014404, 0.16595160961151123, 0.1640637367963791, 0.15558409690856934, 0.15178923308849335, 0.15469633042812347, 0.15122190117835999, 0.1520070731639862, 0.15174877643585205, 0.15355047583580017, 0.21490561962127686, 0.17398352921009064, 0.15784548223018646, 0.15319469571113586, 0.15044008195400238, 0.1600874811410904, 0.15627054870128632, 0.1613706797361374], 'accuracy': [0.9076457619667053, 0.903282642364502, 0.9033365845680237, 0.9063576459884644, 0.8978351950645447, 0.9033552408218384, 0.9074816107749939, 0.9131453633308411, 0.9127440452575684, 0.9073082804679871, 0.9032963514328003, 0.9088945984840393, 0.9118703603744507, 0.8967482447624207, 0.9039113521575928, 0.9069762229919434, 0.9105697870254517, 0.9066866636276245, 0.9078124761581421, 0.911533534526825, 0.9087975025177002, 0.9131419062614441, 0.9115728139877319, 0.9147458672523499, 0.8996348977088928, 0.8965722322463989, 0.9109078645706177, 0.9122697114944458, 0.912509560585022, 0.908207356929779, 0.9124816656112671, 0.9066148996353149, 0.9051694869995117, 0.9114683866500854, 0.9080606698989868, 0.9111766219139099, 0.9131057858467102, 0.911989152431488, 0.9153934717178345, 0.9115018248558044, 0.914294421672821, 0.9138497710227966, 0.8906130194664001, 0.9051879048347473, 0.9132448434829712, 0.9174495935440063, 0.9109888076782227, 0.917506992816925, 0.9092331528663635, 0.9094141721725464]}

EfficientUnet3_7_1_50 = {'loss': [0.6113102436065674, 0.5309041142463684, 0.5014803409576416, 0.49404048919677734, 0.4753141403198242, 0.47655272483825684, 0.4547117054462433, 0.46113312244415283, 0.44050854444503784, 0.4181075990200043, 0.43375492095947266, 0.4177474081516266, 0.40109866857528687, 0.41354072093963623, 0.3943942189216614, 0.3927355110645294, 0.3851621747016907, 0.3819829225540161, 0.3741055727005005, 0.37271150946617126, 0.34932824969291687, 0.34925395250320435, 0.3409675657749176, 0.346507728099823, 0.33110129833221436, 0.3220064043998718, 0.33032888174057007, 0.3195296823978424, 0.3031656742095947, 0.30278557538986206, 0.28997883200645447, 0.29141324758529663, 0.299498587846756, 0.2805066704750061, 0.2906773090362549, 0.28256455063819885, 0.2655814588069916, 0.273570716381073, 0.2718175947666168, 0.26772719621658325, 0.260768324136734, 0.26604869961738586, 0.25781384110450745, 0.250729501247406, 0.2393651157617569, 0.2533656060695648, 0.2522725760936737, 0.24949273467063904, 0.25152987241744995, 0.2294333428144455], 'accuracy': [0.7242468595504761, 0.7535600662231445, 0.7665210366249084, 0.7647228837013245, 0.7794771194458008, 0.7761405110359192, 0.7871397137641907, 0.779278576374054, 0.7914317846298218, 0.805763840675354, 0.7978902459144592, 0.8034029603004456, 0.8102790117263794, 0.8057229518890381, 0.8143131732940674, 0.8173100352287292, 0.8160805106163025, 0.8203320503234863, 0.8252711892127991, 0.8216915726661682, 0.8334119319915771, 0.8358622789382935, 0.8369410037994385, 0.8352493643760681, 0.8418900370597839, 0.8412246108055115, 0.8405135273933411, 0.8464593291282654, 0.8510903120040894, 0.8519267439842224, 0.8552717566490173, 0.8569455742835999, 0.8543322682380676, 0.8643327951431274, 0.862083375453949, 0.8634779453277588, 0.8689538836479187, 0.8675082325935364, 0.8691496253013611, 0.8683980703353882, 0.8692557215690613, 0.8699510097503662, 0.8725457191467285, 0.8779787421226501, 0.8800583481788635, 0.8765053153038025, 0.8728358745574951, 0.8718013763427734, 0.8771407008171082, 0.8849942088127136]}
EfficientUnet3_7_51_100 = {'loss': [0.2347574084997177, 0.23663049936294556, 0.23817101120948792, 0.22790199518203735, 0.23910054564476013, 0.25156474113464355, 0.23816613852977753, 0.23008570075035095, 0.2415759414434433, 0.22116130590438843, 0.23401203751564026, 0.22223559021949768, 0.21462683379650116, 0.22271893918514252, 0.20456244051456451, 0.22064988315105438, 0.21831294894218445, 0.21445821225643158, 0.2180553674697876, 0.217574343085289, 0.20479701459407806, 0.21716530621051788, 0.21889539062976837, 0.20819063484668732, 0.2114972025156021, 0.20253868401050568, 0.19598649442195892, 0.23258762061595917, 0.21258962154388428, 0.24063755571842194, 0.2226238250732422, 0.2049897462129593, 0.20251111686229706, 0.19415231049060822, 0.18528859317302704, 0.20011773705482483, 0.20229721069335938, 0.18554604053497314, 0.20975707471370697, 0.20398616790771484, 0.19507457315921783, 0.18382945656776428, 0.19824838638305664, 0.19611553847789764, 0.18905958533287048, 0.18880079686641693, 0.18358877301216125, 0.23956719040870667, 0.22362884879112244, 0.18815232813358307], 'accuracy': [0.8782510757446289, 0.8780385851860046, 0.8827821612358093, 0.8848880529403687, 0.8793913722038269, 0.8793243169784546, 0.8826867938041687, 0.8861273527145386, 0.880640983581543, 0.8889110684394836, 0.883033275604248, 0.8864269852638245, 0.8891025185585022, 0.8864015340805054, 0.8921500444412231, 0.8849686980247498, 0.8858398199081421, 0.8903261423110962, 0.8880844712257385, 0.8868988752365112, 0.8930051922798157, 0.8885127902030945, 0.887556791305542, 0.8922157883644104, 0.8871785998344421, 0.897110641002655, 0.8942505121231079, 0.8810043334960938, 0.8916049003601074, 0.8858953714370728, 0.8828242421150208, 0.893284261226654, 0.8976603150367737, 0.899143636226654, 0.9022623896598816, 0.8956487774848938, 0.8936826586723328, 0.9015050530433655, 0.8889163136482239, 0.8904185891151428, 0.9005982279777527, 0.900248110294342, 0.8940547108650208, 0.8958621025085449, 0.8968232870101929, 0.9000207185745239, 0.9037230610847473, 0.8798640370368958, 0.8906352519989014, 0.9033562541007996]}
EfficientUnet3_7_101_150 = {'loss': [0.1899532824754715, 0.19003178179264069, 0.19837446510791779, 0.18328121304512024, 0.18770308792591095, 0.19111432135105133, 0.1975233107805252, 0.19011259078979492, 0.18131646513938904, 0.17984722554683685, 0.20558962225914001, 0.17714019119739532, 0.18541987240314484, 0.18385820090770721, 0.1934162676334381, 0.17799049615859985, 0.19537179172039032, 0.17614200711250305, 0.18239393830299377, 0.23231592774391174, 0.19974032044410706, 0.18348217010498047, 0.1703353375196457, 0.18334707617759705, 0.1746504157781601, 0.17911480367183685, 0.1968272179365158, 0.16876763105392456, 0.16372811794281006, 0.16513586044311523, 0.16907361149787903, 0.1623520404100418, 0.20693202316761017, 0.18397453427314758, 0.21593031287193298, 0.17253005504608154, 0.16717803478240967, 0.16959747672080994, 0.17961688339710236, 0.16987352073192596, 0.16586625576019287, 0.16395671665668488, 0.181161567568779, 0.17890708148479462, 0.17414632439613342, 0.1640562117099762, 0.18835847079753876, 0.16582632064819336, 0.16846825182437897, 0.17741085588932037], 'accuracy': [0.8977212905883789, 0.9038723111152649, 0.8952783346176147, 0.9043675661087036, 0.8989711403846741, 0.8994382619857788, 0.8927766680717468, 0.8966510891914368, 0.9009804129600525, 0.9046987295150757, 0.8921558260917664, 0.9030207395553589, 0.9004951119422913, 0.9008197784423828, 0.8992772698402405, 0.9057787656784058, 0.8995546102523804, 0.9058033227920532, 0.9025951623916626, 0.8827939033508301, 0.8990030288696289, 0.9057848453521729, 0.9070664048194885, 0.9027974009513855, 0.9045373797416687, 0.9015806317329407, 0.8978535532951355, 0.9080699682235718, 0.9101574420928955, 0.9067383408546448, 0.9080320000648499, 0.9101988673210144, 0.8923835158348083, 0.9017524123191833, 0.8865968585014343, 0.9055333733558655, 0.9075252413749695, 0.9076837301254272, 0.9062215685844421, 0.9061614274978638, 0.9062895774841309, 0.9102596640586853, 0.8998277187347412, 0.9054696559906006, 0.9085239171981812, 0.9083424210548401, 0.8964505791664124, 0.906798243522644, 0.9062511920928955, 0.9031941294670105]}
EfficientUnet3_7_151_200 = {'loss': [0.16995449364185333, 0.18466529250144958, 0.1777454912662506, 0.16426050662994385, 0.16659985482692719, 0.15870146453380585, 0.1630038321018219, 0.16469047963619232, 0.15979436039924622, 0.1746634989976883, 0.15806196630001068, 0.15724413096904755, 0.1605886071920395, 0.17311087250709534, 0.156727135181427, 0.15982596576213837, 0.15402983129024506, 0.15345284342765808, 0.1519506722688675, 0.15251393616199493, 0.15412665903568268, 0.20104221999645233, 0.1828247606754303, 0.17125621438026428, 0.19881296157836914, 0.1760208159685135, 0.16140015423297882, 0.15634894371032715, 0.19093188643455505, 0.15843693912029266, 0.18822075426578522, 0.15810969471931458, 0.15680648386478424, 0.15454202890396118, 0.16255971789360046, 0.15520673990249634, 0.15374314785003662, 0.1506282091140747, 0.15283116698265076, 0.16493158042430878, 0.1640384942293167, 0.15156516432762146, 0.1535646766424179, 0.15123358368873596, 0.15945598483085632, 0.17016597092151642, 0.1600845754146576, 0.15850569307804108, 0.16207392513751984, 0.15532805025577545], 'accuracy': [0.9087920784950256, 0.8999646902084351, 0.9001814126968384, 0.9070799946784973, 0.9114982485771179, 0.9089385271072388, 0.9095039963722229, 0.906104564666748, 0.9093808531761169, 0.9066191911697388, 0.9124276638031006, 0.9121708869934082, 0.9122194647789001, 0.9077408313751221, 0.9146954417228699, 0.9101917147636414, 0.9127465486526489, 0.9137989282608032, 0.9122942090034485, 0.9126355648040771, 0.910522997379303, 0.8980979919433594, 0.9027479290962219, 0.9064105153083801, 0.9025622010231018, 0.9061833620071411, 0.9094488620758057, 0.9124819040298462, 0.9020212292671204, 0.9110623002052307, 0.9029123783111572, 0.9088098406791687, 0.9104154109954834, 0.9159621596336365, 0.9095718264579773, 0.9101879596710205, 0.9113961458206177, 0.9163849949836731, 0.909877598285675, 0.906809389591217, 0.9078702926635742, 0.9132732152938843, 0.9150758981704712, 0.9142982363700867, 0.9079982042312622, 0.9109975695610046, 0.9116770625114441, 0.9115605354309082, 0.9110787510871887, 0.9124309420585632]}

# process
EfficientUnet_loss, EfficientUnet_accuracy = [], []
EfficientUnet_loss += EfficientUnet1_50['loss'] + EfficientUnet51_100['loss'] + EfficientUnet101_150['loss'] + EfficientUnet151_200['loss']
EfficientUnet_accuracy += EfficientUnet1_50['accuracy'] + EfficientUnet51_100['accuracy'] + EfficientUnet101_150['accuracy'] + EfficientUnet151_200['accuracy']

EfficientUnet3_5_loss, EfficientUnet3_5_accuracy = [], []
EfficientUnet3_5_loss += EfficientUnet3_5_1_50['loss'] + EfficientUnet3_5_51_100['loss'] + EfficientUnet3_5_101_150['loss'] + EfficientUnet3_5_151_200['loss']
EfficientUnet3_5_accuracy += EfficientUnet3_5_1_50['accuracy'] + EfficientUnet3_5_51_100['accuracy'] + EfficientUnet3_5_101_150['accuracy'] + EfficientUnet3_5_151_200['accuracy']

EfficientUnet3_7_loss, EfficientUnet3_7_accuracy = [], []
EfficientUnet3_7_loss += EfficientUnet3_7_1_50['loss'] + EfficientUnet3_7_51_100['loss'] + EfficientUnet3_7_101_150['loss'] + EfficientUnet3_7_151_200['loss']
EfficientUnet3_7_accuracy += EfficientUnet3_7_1_50['accuracy'] + EfficientUnet3_7_51_100['accuracy'] + EfficientUnet3_7_101_150['accuracy'] + EfficientUnet3_7_151_200['accuracy']

# 创建一个新的图像窗口，并设置子图布局为1行2列
fig, axs = plt.subplots(1, 2)

# 第一个子图：Loss
axs[0].plot(range(len(EfficientUnet_loss)), EfficientUnet_loss, label='Effi-Att-Unet loss')
axs[0].plot(range(len(EfficientUnet3_5_loss)), EfficientUnet3_5_loss, label='Effi-Att-Unet3+(5 layers) loss')
axs[0].plot(range(len(EfficientUnet3_7_loss)), EfficientUnet3_7_loss, label='Effi-Att-Unet3+(7 layers) loss')
axs[0].set_title('Loss Decline Curve')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()

# 第二个子图：Accuracy
axs[1].plot(range(len(EfficientUnet_accuracy)), EfficientUnet_accuracy, label='Effi-Att-Unet accuracy')
axs[1].plot(range(len(EfficientUnet3_5_accuracy)), EfficientUnet3_5_accuracy, label='Effi-Att-Unet3+(5 layers) accuracy')
axs[1].plot(range(len(EfficientUnet3_7_accuracy)), EfficientUnet3_7_accuracy, label='Effi-Att-Unet3+(7 layers) accuracy')
axs[1].set_title('Accuracy Rising Curve')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].legend()

# 调整子图之间的间距
plt.tight_layout()

# 显示图像
plt.show()