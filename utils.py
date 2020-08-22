from sklearn.model_selection import train_test_split
import numpy as np

KHCONST = list(u'កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអឣឤឥឦឧឨឩឪឫឬឭឮឯឰឱឲឳ')
KHVOWEL = list(u'឴឵ាិីឹឺុូួើឿៀេែៃោៅ\u17c6\u17c7\u17c8')
# subscript, diacritics
KHSUB = list(u'្')
KHDIAC = list(u"\u17c9\u17ca\u17cb\u17cc\u17cd\u17ce\u17cf\u17d0") #MUUSIKATOAN, TRIISAP, BANTOC,ROBAT,
KHSYM = list('៕។៛ៗ៚៙៘,.? ') # add space
KHNUMBER = list(u'០១២៣៤៥៦៧៨៩0123456789') # remove 0123456789
# lunar date:  U+19E0 to U+19FF ᧠...᧿
KHLUNAR = list('᧠᧡᧢᧣᧤᧥᧦᧧᧨᧩᧪᧫᧬᧭᧮᧯᧰᧱᧲᧳᧴᧵᧶᧷᧸᧹᧺᧻᧼᧽᧾᧿')

def pad_input(sents, seq_len, isFeature = True):
    features = np.zeros((len(sents), seq_len),dtype=int)
    if isFeature == False:
        features +=-1
    for ii, review in enumerate(sents):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

def split_data(X_char, y_char,chars2idx,sentence_length=100):
    X_train_char, X_test_char, y_train_char, y_test_char = train_test_split(X_char, y_char, test_size=0.20, random_state=1)
    print("len X_train_char:", len(X_train_char), " data:", len(X_train_char[0]))
    print("len y_train_char:", len(y_train_char), " data:", len(y_train_char[0]))
    print("len X_test_char:", len(X_test_char), " data:", len(X_test_char[0]))
    print("len y_test_char:", len(y_test_char), " data:", len(y_test_char[0]))

    for i, sentence in enumerate(X_train_char):
        # Looking up the mapping dictionary and assigning the index to the respective words
        X_train_char[i] = [chars2idx[c] if c in chars2idx else 1 for c in sentence]

    for i, sentence in enumerate(X_test_char):
        # For test sentences, we have to tokenize the sentences as well
        X_test_char[i] = [chars2idx[c] if c in chars2idx else 1 for c in sentence]

    X_train_char = pad_input(X_train_char,sentence_length)
    X_test_char = pad_input(X_test_char,sentence_length)
    y_train_char = pad_input(y_train_char,sentence_length,False)
    y_test_char = pad_input(y_test_char,sentence_length,False)

    print("Train",len(X_train_char), "X_train_char[0]", X_train_char[0])
    print("Test",len(X_test_char), "X_test_char[0]", X_test_char[0])

    return X_train_char, X_test_char, y_train_char, y_test_char 

def one_hot_encode(arr, n_labels):
    
    # Initialize the the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot

def count_correct_word(correctstr, predictionstr):
  #print("prediction:", prediction)
  #print("   correct:", correct)
  correct = [int(c) for c in correctstr]
  prediction = [int(p) for p in predictionstr]
  correct_count = 0
  for i,c in enumerate(correct):
    p = prediction[i]
    if c==1 and p==1:
      correct_count += 1

  return correct_count

def get_correction_list():
  fixes=[]
  #fix bad segmented data - especially proper noun
  fixes.append(['អា​ខោន​','អាខោន​']) #account 
  fixes.append(["​ឱវា​ទ","​ឱវាទ"])
  fixes.append(["នៅ​សប្តា​ហ៏​","នៅ​សប្តាហ៏​"])
  fixes.append(['ស៊ុន​ ​ចាន់​ថុល','ស៊ុន​ ​ចាន់ថុល'])
  fixes.append(['សាយ​ ​មក​រា','សាយ​ ​មករា'])
  fixes.append(['រោង​កុន','រោងកុន'])
  fixes.append(['ដេី​មជ្រៃ','ដេីមជ្រៃ'])
  fixes.append(['​ដេី​ម​','​ដេីម​'])
  fixes.append(['មន្ទី​ពេទ្យ','មន្ទីពេទ្យ'])
  fixes.append(['ជាដរាបត​ទៅ','ជា​ដរាប​តទៅ'])
  fixes.append(['ដេី​ម្បី​','ដេីម្បី​'])
  fixes.append(['ធ្វេី​ការកាត់​យក','ធ្វេីការ​កាត់​យក']) # interesting!!!!
  fixes.append(['ភូមិ​ផ្សារ​ដី​ហុយ','ភូមិ​ផ្សារដីហុយ'])
  fixes.append(['សុខ​ ​ពេញ​វុធ','សុខ​ ​ពេញវុធ'])
  fixes.append(['ប្រតិបត្តិការបង្ក្រាប','ប្រតិបត្តិការ​បង្ក្រាប'])
  fixes.append(['health​.​com​.​kh','health.com.kh'])
  fixes.append(['ពួយ​ ​ច័ន្ទ​សុគុណ','ពួយ​ ​ច័ន្ទសុគុណ'])
  fixes.append(['ភូមិ​ ​ថ្ម​ដា','ភូមិ​ ​ថ្មដា'])
  fixes.append(['សង្កាត់​ ​បឹង​កន្សែង','សង្កាត់​ ​បឹងកន្សែង']) # TODO what is orig text???
  fixes.append(['ក្រុង​ ​បាន​លុង','ក្រុង​ ​បានលុង'])
  fixes.append(['នុត​ ​សុគន្ធ​ ​ផាន់​នី','នុត​ ​សុគន្ធ​ ​ផាន់នី'])
  #fixes.append(['ព្រះរាជាណា ច្រក កម្ពុជា','ព្រះរាជាណាច្រ ក កម្ពុជា']) #bad spelling – no better seg
  fixes.append(['កាក​បាត​ក្រហម','កាកបាត​ក្រហម']) # from https://www.redcross.org.kh/ កាកបាទក្រហមកម្ពុជា
  fixes.append(['ស្រុក​ ​កំពង់ត្រា​ច','ស្រុក​ ​កំពង់ត្រាច'])
  fixes.append(['ឃុំ​ ​ថ្ម​កែវ','ឃុំ​ ​ថ្មកែវ'])
  fixes.append(['ស្រុក​ ​អង្គរ​ជ័យ','ស្រុក​ ​អង្គរជ័យ'])
  fixes.append(['កោះ​តា​កូវ ឃុំ​ឬ​ស្សី','កោះ​តាកូវ ឃុំ​ឬស្សី'])
  fixes.append(['ម៉ាក​ ​សែន​សូនីតា','ម៉ាក​ ​សែនសូនីតា'])
  fixes.append(['ជា​ ​ដេី​ម​ ​នោះ','ជា​ ​ដេីម​ ​នោះ'])
  fixes.append(['ដេី​ម្បី​ ​ជ្រាប','ដេីម្បី​ ​ជ្រាប']) #prob eiy is two chars េ ⁣ី instead of one char
  fixes.append(['ក​រករណី','ករ​ករណី'])
  fixes.append(['ទោ​ចក្រ​យានយន្ត','ទោចក្រយានយន្ត'])
  fixes.append(['ឃុំ​ ​អន្លង់​វិល','ឃុំ​ ​អន្លង់វិល'])
  fixes.append(['​ពោធ៍​សាត់','​ពោធ៍សាត់'])
  fixes.append(['ណាក់​ ​ស្រី​ណា','ណាក់​ ​ស្រីណា'])
  fixes.append(['ពិធីការ​នីមួយ','ពិធីការនី​មួយ']) #valid both ways, need context
  fixes.append(['តូរុន​តូ','តូរុនតូ']) #Toronto
  fixes.append(['ដាយ​ណូស័រ','ដាយណូស័រ']) #Dinosaur
  fixes.append(['ហ្គារ៉ាល់​ហូស','ហ្គារ៉ាល់ហូស']) #Guarulhos in Sao Paulo
  fixes.append(['ខេមរៈ​ ​សិរី​មន្ត','ខេមរៈ​ ​សិរីមន្ត'])#ខេមរៈ \u200bសិរី\u200bមន្ត
  fixes.append(['ឱម​ ​យ៉ិន​ទៀង','ឱម​ ​យ៉ិនទៀង'])
  fixes.append(['មហា​លាប','មហាលាប'])
  fixes.append(['ការចាប់អារម្ម​ណ៏​','ការចាប់អារម្មណ៏​'])
  fixes.append(['វ៉ិច​ទ័រ​','វ៉ិចទ័រ​']) #rector
  fixes.append(['សុវណ្ណ​ ​ឬ​ទ្ធី','សុវណ្ណ​ ​ឬទ្ធី'])
  fixes.append(['ខាន់​ ​ចាន់​សុផល','ខាន់​ ​ចាន់សុផល'])
  fixes.append(['ទំនួលខុសត្រូវ​','ទំនួល​ខុសត្រូវ​'])
  fixes.append(['ឯ​ណេះ','ឯណេះ']) #confirm with Chuon Nath Dict
  fixes.append(['ហេយ៍​វ៉ិន','ហេយ៍វ៉ិន']) #heaven
  fixes.append(['សុខ​ ​ពេញ​វុធ','សុខ​ ​ពេញវុធ'])
  fixes.append(['វ៉ុល​កា','វ៉ុលកា']) #vulgar
  fixes.append(['ជា​ម​ ​ហុី​ម','ជាម​ ​ហុីម'])
  fixes.append(['​ការ​ប្រកាស​','​ការប្រកាស​'])
  fixes.append(['អនុ​ប្រធាន','អនុប្រធាន'])
  fixes.append(['កិច្ចខិតខំប្រឹងប្រែង','កិច្ច​ខិតខំ​ប្រឹងប្រែង']) #???
  fixes.append(['សស្អាត','ស​ស្អាត'])
  fixes.append(['ហាន់​ ​ជី​អ៊ុន','ហាន់​ ​ជីអ៊ុន'])
  fixes.append(['ស៊ុន​ ​ចាន់​ថុល','ស៊ុន​ ​ចាន់ថុល'])
  fixes.append(['លីន​ដា​','លីនដា​'])
  fixes.append(['យានយន្តបន្ទាប់ពី','យានយន្ត​បន្ទាប់ពី'])
  fixes.append(['ព្រៃ​ស្អាក​ស្រុក​អន្លង់​វែង','ព្រៃស្អាក​ស្រុក​អន្លង់វែង'])
  fixes.append(['​ប៉ា​កុង​','​ប៉ាកុង​']) # first 50 articles
  fixes.append(['ឈើ​ ​អុស​ក្រាក់','ឈើ​ ​អុសក្រាក់'])
  fixes.append(['រតនៈ​គីរី​','រតនៈគីរី​'])
  fixes.append(['ឯក​ឧត្តម','ឯកឧត្តម'])
  fixes.append(['ឱម​ ​យិន​ទៀង','ឱម​ ​យិនទៀង'])
  fixes.append(['​ចំដែនដី​','​ចំ​ដែនដី​']) # orig has space after រៀប
  fixes.append(['ឱវា​ទ','ឱវាទ']) # correct spelling
  fixes.append(['អ៊ិន​វេស​មិន','អ៊ិនវេសមិន']) #investment
  fixes.append(['សហគម​ន៏','សហគមន៏'])
  fixes.append(['​គ្រូ​សារ​','​គ្រូសារ​']) # orig has bad ub, mispelled but should be removed before seg
  fixes.append(['សុផុនពេល​សួរ','សុផុន​ពេល​សួរ'])
  fixes.append(['ស្រុក​កែវ​សីម៉ា','ស្រុក​កែវសីម៉ា'])
  fixes.append(['ហាន់​ជ័យ','ហាន់ជ័យ'])
  fixes.append(['បូរី​ ​ស​ចនសុន','បូរីស​ ​ចនសុន'])
  fixes.append(['បូរី​ ​ស​ចន​សុន','បូរីស​ ​ចនសុន'])
  fixes.append(['បូរីស​ ​ចន​សុន','បូរីស​ ​ចនសុន'])
  fixes.append(['អំណ​រ​គុណ','អំណរ​គុណ']) # bad orig
  fixes.append(['ផលប៉ះពាល់​','ផល​ប៉ះពាល់​']) # ???
  fixes.append(['អ្នកវិនិយោគ​','អ្នក​វិនិយោគ​']) #???
  fixes.append(['ភូមិ​ ​ស្រះ​ជ្រៃ​','ភូមិ​ ​ស្រះជ្រៃ​'])
  fixes.append(['ឃុំ​ ​បន្ទាយ​ឆ្មា​','ឃុំ​ ​បន្ទាយឆ្មា​'])
  fixes.append(['ស្រុក​ ​ថ្ម​ពួក​','ស្រុក​ ​ថ្មពួក​'])
  fixes.append(['ទ្វីត​ធឺ','ទ្វីតធឺ']) #twitter
  fixes.append(['​ធំ​ដុំ','​ធំដុំ'])
  fixes.append(['​រួមមាន​','​រួម​មាន​']) #?
  fixes.append(['គុយ​វ៉ែត','គុយវ៉ែត'])
  fixes.append(['ងងុយគេង','ងងុយ​គេង'])
  fixes.append(['ស​ម​ ​រង្ស៊ី','សម​ ​រង្ស៊ី'])
  fixes.append(['សោ​ ​ចាន់​ដេត','សោ​ ​ចាន់ដេត'])
  fixes.append(['ជាលាយលក្ខណ៍អក្សរ','ជា​លាយលក្ខណ៍​អក្សរ'])
  fixes.append(['ផលិតកម្មវិធី','ផលិតកម្ម​វិធី'])
  fixes.append(['​សាជាថ្មី​','​សា​ជា​ថ្មី​']) # ???
  fixes.append(['ឧកញ៉ា​ស្រី​ ​ចាន់​ថន','ឧកញ៉ា​ ​ស្រី​ ​ចាន់ថន'])
  fixes.append(['ភូមិ​ ​ព្រៃ​ល្វា','ភូមិ​ ​ព្រៃល្វា'])
  fixes.append(['សង្កាត់​ ​ចោម​ចៅ ','សង្កាត់​ ​ចោមចៅ'])
  fixes.append(['ខណ្ឌ​ ​ពោធិ៍​សែន','ខណ្ឌ​ ​ពោធិ៍សែន'])
  fixes.append(['ឃុំ​ ​ព្រែក​តា​មាក់','ឃុំ​ ​ព្រែកតាមាក់'])
  fixes.append(['​បាត់បង្ក','​បាត់​បង្ក'])
  fixes.append(['កើតមានឡើង','កើត​មាន​ឡើង'])
  fixes.append(['លូក​លាន់','លូកលាន់'])
  fixes.append(['លោក​ ​ជិន​ ​ម៉ាលី​ន','លោក​ ​ជិន​ ​ម៉ាលីន'])
  fixes.append(['អ្នករាយការណ៍','អ្នក​រាយការណ៍'])
  fixes.append(['គង់​ ​រ៉ៃ​យ៉ា','គង់​ ​រ៉ៃយ៉ា'])
  fixes.append(['លោក​ ​សួង​ ​នាគ​ព័ន្ធ','លោក​ ​សួង​ ​នាគព័ន្ធ'])
  fixes.append(['ធ្វើទុក្ខបុកម្នេញ','ធ្វើ​ទុក្ខបុកម្នេញ'])  # doc 75 mark
  fixes.append(['បោះបង់ចោល','បោះបង់​ចោល'])
  fixes.append(['ខ្លួនឯង','ខ្លួន​ឯង']) # ???
  fixes.append(['សប្បាយចិត្ត','សប្បាយ​ចិត្ត'])
  fixes.append(['មើលទៅ','មើល​ទៅ']) # ??? if
  fixes.append(['សោយសុខ','សោយ​សុខ'])
  fixes.append(['ទៅលើ','ទៅ​លើ'])
  fixes.append(['ឃុំខ្លួន','ឃុំ​ខ្លួន'])
  fixes.append(['ជាប់ឃុំ','ជាប់​ឃុំ'])
  fixes.append(['លោក​ ​ហេង​ ​ដូន​នី','លោក​ ​ហេង​ ​ដូននី'])
  fixes.append(['លោក​ ​កែម​ ​គិម​ស្រន់','លោក​ ​កែម​ ​គិមស្រន់'])
  fixes.append(['ស្រុក​ ​វាល​វែង','ស្រុក​ ​វាលវែង'])
  fixes.append(['ចោទប្រកាន់តែ','ចោទប្រកាន់​តែ']) # doc 80 mark
  fixes.append(['ឈិញ​ ​ស៊ី​ថា','ឈិញ​ ​ស៊ីថា'])
  fixes.append(['ដាក់ពាក្យបណ្តឹង','ដាក់​ពាក្យ​បណ្តឹង'])
  fixes.append(['ម៉ាក​ ​ណូ​គា','ម៉ាក​ ​ណូគា'])
  fixes.append(['ធូ​ ​ស្រី​ទូច','ធូ​ ​ស្រីទូច'])
  fixes.append(['ម៉ូតូកង់​បី','ម៉ូតូ​កង់​បី'])
  fixes.append(['ឃុំ មង់​រៀវ','ឃុំ មង់រៀវ'])
  fixes.append(['លោក​ ​ជា​ ​ចាន់​តូ','លោក​ ​ជា​ ​ចាន់តូ'])
  fixes.append(['ស៊ែ​ស្វី​ច','ស៊ែ​ស្វីច']) #shared switch
  fixes.append(['ប្រតិបត្តិការផ្ទេរ','ប្រតិបត្តិការ​ផ្ទេរ'])
  fixes.append(['បញ្ជាការស្រាល','បញ្ជាការ​ស្រាល'])
  fixes.append(['F​-​16','F-16'])
  fixes.append(['C​-​17','C-17'])
  fixes.append(['គោលនយោបាយ','គោល​នយោបាយ']) # id 90 mark
  fixes.append(['ហាន​ ​សុខ​ន','ហាន​ ​សុខន'])
  fixes.append(['ខៀវ​ ​ទេ​ព','ខៀវ​ ​ទេព'])
  fixes.append(['កាញារី​ទ្ធ','កាញារីទ្ធ'])
  fixes.append(['ជី​ហែ','ជីហែ']) # ផ្សារជីហែ
  fixes.append(['វិទ្យាល័យ​ ​ជី​ហែរ','វិទ្យាល័យ​ ​ជីហែរ'])
  fixes.append(['ស្រុកោះ​កោះ​សូទិន','ស្រុកោះ​កោះសូទិន'])
  fixes.append(['ជំរុញឲ្យ','ជំរុញ​ឲ្យ'])
  fixes.append(['ការបាក់ទឹកចិត្ត','ការបាក់​ទឹកចិត្ត'])
  fixes.append(['ហង់ ជួន​ណារីតា','ហង់ ជួនណារីតា'])
  fixes.append(['ជាទីមោទនៈ','ជា​ទី​មោទនៈ'])
  fixes.append(['សឿ សុជា​តា','សឿ សុជាតា'])
  fixes.append(['ហង់​ ​ជួន​ណារ៉ុន','ហង់​ ​ជួនណារ៉ុន'])
  fixes.append(['ក​សាង​ធនធាន','កសាង​ធនធាន'])
  fixes.append(['ស្នង​ឬ​ស្សី','ស្នង​ឬស្សី'])
  fixes.append(['ខៀវ​ ​កាញារី​ទ្ធ','ខៀវ​ ​កាញារីទ្ធ'])
  fixes.append(['ត្រេន​ ​ដី​ង','ត្រេនដីង']) #trending
  fixes.append(['ទៅវិញទៅមក','ទៅ​វិញ​ទៅ​មក']) # ??
  fixes.append(['ក្ដី​ស្រមៃ','ក្ដីស្រមៃ']) #prob mispell jerg da
  fixes.append(['ឬ​ស្សីកែវ','ឬស្សីកែវ'])
  fixes.append(['អេង​ ​សុវណ្ណ​តារា','អេង​ ​សុវណ្ណតារា'])
  fixes.append(['កុង​ទីន​រ័','កុងទីនរ័'])
  fixes.append(['ជាបន្តបន្ទាប់','ជា​បន្តបន្ទាប់'])
  fixes.append(['ចលត័ជាប្រចាំ','ចលត័​ជា​ប្រចាំ'])
  fixes.append(['ឃុំ​ដី​ឥដ្ឋ','ឃុំ​ដីឥដ្ឋ'])
  fixes.append(['វី​ដែ​អូ','វីដែអូ']) #misspell វីដេអូ 
  fixes.append(['ត្រី​ខ​','ត្រីខ​'])
  fixes.append(['ទឹក​ត្រី​','ទឹកត្រី​'])
  fixes.append(['បង្ខំឲ្យ','បង្ខំ​ឲ្យ']) # first hundred docs
  fixes.append(['ស្ដុក\u200bប្រវឹក','ស្ដុកប្រវឹក'])
  fixes.append(['វាល\u200bរិញ\u200b','វាលរិញ\u200b'])
  fixes.append(['ណៃ\u200b \u200bវង្ស\u200bដា','ណៃ\u200b \u200bវង្សដា'])
  fixes.append(['លោក\u200b \u200bឃ\u200bន \u200bជឺ ','លោក\u200b \u200bឃន\u200b \u200bជឺ '])
  fixes.append(['បុត\u200b \u200bចាន់\u200bណា','បុត\u200b \u200bចាន់ណា'])
  fixes.append(['ឯ\u200bម\u200b \u200bច័ន្ទ\u200bមក\u200bរា','ឯម\u200b \u200bច័ន្ទមករា'])
  fixes.append(['សង់\u200bដ្រី\u200bន\u200b \u200bឌុយ\u200bរី','សង់ដ្រីន\u200b \u200bឌុយរី'])
  fixes.append(['សុខ\u200b \u200bពេញ\u200bវុធ','សុខ\u200b \u200bពេញវុធ'])
  fixes.append(['សោម \u200bពុទ្ធ\u200bតារា','សោម \u200bពុទ្ធតារា'])
  fixes.append(['ម៉ៅ\u200b ច័ន្ទ\u200bមធុរិទ្ធ','ម៉ៅ\u200b ច័ន្ទមធុរិទ្ធ'])
  fixes.append(['បូពិន្ទ\u200b ណា\u200bរ័ត្ន','បូពិន្ទ\u200b ណារ័ត្ន'])
  fixes.append(['ជា\u200b \u200bប៊ុន\u200bហេង','ជា\u200b \u200bប៊ុនហេង'])
  fixes.append(['ណារិក\u200bដ្រា\u200b \u200bមូឌី','ណារិកដ្រា\u200b \u200bមូឌី'])
  fixes.append(['ស្រុក\u200bព្រៃ\u200bនប់','ស្រុក\u200bព្រៃនប់'])
  fixes.append(['ស្រុក\u200bព្រៃ\u200bនប់','ស្រុក\u200bព្រៃនប់'])
  fixes.append(['ខណ្ឌ\u200bចំការ\u200bមន','ខណ្ឌ\u200bចំការមន'])
  fixes.append(['ខណ្ឌ\u200bពោធិ៍\u200bសែន\u200bជ័យ','ខណ្ឌ\u200bពោធិ៍សែនជ័យ'])
  fixes.append(['សង្កាត់\u200bទឹក\u200bល្អក់','សង្កាត់\u200bទឹកល្អក់'])
  fixes.append(['សង្កាត់\u200bផ្សារ\u200bថ្មី','សង្កាត់\u200bផ្សារថ្មី'])
  fixes.append(['សង្កាត់\u200bជ្រោយ\u200bចង្វារ','សង្កាត់\u200bជ្រោយចង្វារ'])
  fixes.append(['សង្កាត់\u200bវាល\u200bវង់','សង្កាត់\u200bវាលវង់'])
  fixes.append(['សង្កាត់\u200bកោះ\u200bដាច់','សង្កាត់\u200bកោះដាច់'])
  fixes.append(['សង្កាត់\u200bចាក់អង្រែ\u200bលើ','សង្កាត់\u200bចាក់អង្រែលើ'])
  fixes.append(['សង្កាត់\u200bទន្លេ\u200bបាសាក់','សង្កាត់\u200bទន្លេបាសាក់'])
  fixes.append(['ឃុំ\u200bត្រពាំង\u200bជោ','ឃុំ\u200bត្រពាំងជោ'])
  fixes.append(['ឃុំ\u200bក្រាំង\u200bល្វា','ឃុំ\u200bក្រាំងល្វា'])
  fixes.append(['ឃុំ\u200bដូន\u200bសរ','ឃុំ\u200bដូនសរ'])
  fixes.append(['ឃុំ\u200bអូរ\u200bតាប៉ោង','ឃុំ\u200bអូរតាប៉ោង'])
  fixes.append(['ឃុំ\u200bក្បាល\u200bដំរី','ឃុំ\u200bក្បាលដំរី'])
  fixes.append(['ឃុំ\u200bត្រពាំង\u200bគង','ឃុំត្រពាំងគង'])
  fixes.append(['ភូមិ\u200bច្រក\u200bទៀក','ភូមិ\u200bច្រកទៀក'])
  fixes.append(['ភូមិ\u200bខ្សាច់\u200bស','ភូមិ\u200bខ្សាច់ស'])
  fixes.append(['ភូមិ\u200bអូរ\u200bត្រូន','ភូមិ\u200bអូរត្រូន'])
  fixes.append(['ភូមិ\u200bថ្ម\u200bធំ','ភូមិ\u200bថ្មធំ'])
  fixes.append(['ភូមិ\u200bចំការ\u200bដូង','ភូមិ\u200bចំការដូង'])
  fixes.append(['ភូមិ\u200bព្រៃ\u200bផ្តៅ','ភូមិ\u200bព្រៃផ្តៅ'])
  fixes.append(['ភូមិ\u200bបឹង\u200bវែង','ភូមិ\u200bបឹងវែង'])
  fixes.append(['\u200b \u200b.\u200b \u200bcom\u200b','.com\u200b'])
  fixes.append(['ចូ\u200b \u200bបាយ\u200bដិន','ចូ\u200b \u200bបាយដិន'])
  fixes.append(['\u200bណាន់\u200bស៊ី\u200b','\u200bណាន់ស៊ី\u200b'])
  fixes.append(['លោក\u200bត្រាំតទៅទៀត','លោក\u200bត្រាំ\u200bតទៅទៀត'])

  #fixes.append(['','']) #u200bប៊ុល\u200bហ្គារី \u200b,  \u200bហុង\u200bគ្រី  Hungary
  return fixes

def correct_str(str):
  for f in get_correction_list():
    str = str.replace(f[0], f[1])
  return str

def cleanup_str(str):
  str = str.strip('\u200b').strip()
  str = str.replace("  ", " ") # clean up 2 spaces to 1
  str = str.replace(" ", "\u200b")   # ensure 200b around space
  # clean up
  str = str.replace("\u200b\u200b", '\u200b')   # clean up dupe 200b
  str = str.replace("\u200b\u200b", '\u200b')   # in case multiple
  str = correct_str(str) # assume space has 200b wrapped around
  
  # remove special characters
  str = str.replace(u"\u2028", "") # line separator
  str = str.replace(u"\u200a", "")  # hair space
  str = str.strip().replace('\n','').replace('  ',' ')
  return str

# character base segmentation
def seg_char(str_sentence):
  #str_sentence = str_sentence.replace(u'\u200b','')
  segs = []
  for phr in str_sentence.split('\u200b'):
      #phr_char = phr.replace(' ','')
      for c in phr:
          segs.append(c)
  return segs

def is_khmer_char(ch):
  if (ch >= '\u1780') and (ch <= '\u17ff'): return True
  if ch in KHSYM: return True
  if ch in KHLUNAR: return True
  return False

def is_start_of_kcc(ch):
  if is_khmer_char(ch):
    if ch in KHCONST: return True
    if ch in KHSYM: return True
    if ch in KHNUMBER: return True
    if ch in KHLUNAR: return True
    return False
  return True

def seg_kcc(str_sentence):
    segs = []
    cur = ""
    sentence = str_sentence
    #for phr in str_sentence.split(): #no longer split by space, use 200b
    #    print("phr: '", phr,"'")
    for word in sentence.split('\u200b'):
      #print("PHR:[%s] len:%d" %(phr, len(phr)))
      for i,c in enumerate(word):
          #print(i," c:", c)
          cur += c
          nextchar = word[i+1] if (i+1 < len(word)) else ""
          
          # cluster non-khmer chars together
          if not is_khmer_char(c) and nextchar != " " and nextchar != "" and not is_khmer_char(nextchar): 
            continue
          # cluster number together
          if c in KHNUMBER and nextchar in KHNUMBER: 
            continue
            
          # cluster non-khmer together
          # non-khmer character has no cluster
          if not is_khmer_char(c) or nextchar==" " or nextchar=="":
              segs.append(cur)
              cur=""
          elif is_start_of_kcc(nextchar) and not (c in KHSUB):
              segs.append(cur)
              cur="" 
        # add space back after split
        #segs.append(" ")   
    return segs # [:-1] # trim last space

def preprocess(input_str,model):
  t = cleanup_str(input_str)
  skcc = seg_kcc(t)
  #print("len kcc:", len(skcc), skcc)
  x = [model.kccs2int[c] if c in model.kccs2int else 1 for c in skcc]
  return x, skcc

def postprocess(pred,skcc,sep=" "):
  separator = "-"
  tkcc = []
  for k in skcc:
    tkcc.append(k)
  #print("kcc:", tkcc)
  complete = ""
  for i, p in enumerate(pred):
      if p == 1.:
        complete += separator + tkcc[i]
      else:
        complete += tkcc[i]
  complete = complete.strip(separator)
  complete = complete.replace(separator+" "+separator, " ")
  return complete