
n_channel = 16
overlap = 3.1875
frame_size = 2048
hop = 256
sf = 32000
frame_sample_count = 16000

local = 9999
event = 'ALARM'

bag_path = f'/home/pierre-olivier/catkin_ws/src/bag/article/{local}/'
dict_path = f'/home/pierre-olivier/catkin_ws/src/egonoise/src/database_{local}/'

if local == 2008:
    list_bag_noise = ['AL5', 'AL6', 'AL7', 'AL8', 'AL15', 'AL16', 'AL17', 'AL18']
    list_bag_database = ['AL1', 'AL2', 'AL3', 'AL11', 'AL12', 'AL13']
    bg_intensity = 3000
    gain = 1.0

elif local == 1004:
    list_bag_noise = ['AL3', 'AL4', 'AL5', 'AL12', 'AL14', 'AL22', 'AL23', 'AL25']
    list_bag_database = ['AL1', 'AL2', 'AL11', 'AL15', 'AL21', 'AL24']
    bg_intensity = 1000
    gain = 1.0

elif local == 9999:
    list_bag_noise = ['AL3', 'AL4', 'AL5', 'AL12', 'AL14', 'AL22', 'AL23', 'AL25']
    list_bag_database = ['AL1']#, 'AL2', 'AL11', 'AL15', 'AL21', 'AL24']
    bg_intensity = 1000
    gain = 1.0

elif local == 3000:
    list_bag_noise = ['AL2', 'AL4', 'AL6', 'AL7', 'AL13', 'AL15', 'AL17', 'AL18']
    list_bag_database = ['AL1', 'AL3', 'AL5', 'AL12', 'AL14', 'AL16']
    bg_intensity = 1500
    gain = 1.0

elif local == 'video':
    list_bag_noise = ['VID2']
    list_bag_target = [['VID2', 1]]
    list_bag_database = ['CALIB']
    bg_intensity = 0
    gain = 0.0

if event == 'DOOR':
    list_bag_target = [
        ['DOOR1', 1],
        ['DOOR2', 1],
        ['DOOR3', 1],
        ['DOOR4', 1],
        ['DOOR5', 1],
        ['DOOR6', 1],
        ['DOOR7', 1],
        ['DOOR8', 1],
        ['DOOR9', 1],
        ['DOOR10', 1],
        ['DOOR11', 1],
        ['DOOR12', 1],
        ['DOOR13', 1],
        ['DOOR14', 1],
        ['DOOR15', 1],
    ]
elif event=='MUSIC':
    gain = 2.0
    list_bag_target = [
        ['MUSIC1', 1],
        ['MUSIC2', 1],
        ['MUSIC3', 1],
        ['MUSIC4', 1],
        ['MUSIC5', 1],
        ['MUSIC6', 1],
        ['MUSIC7', 1],
        ['MUSIC8', 1],
        ['MUSIC9', 1],
        ['MUSIC10', 1],
        ['MUSIC11', 1],
        ['MUSIC12', 1],
        ['MUSIC13', 1],
        ['MUSIC14', 1],
        ['MUSIC15', 1],
    ]
elif event=='ALARM':
    list_bag_target = [
        ['ALARM1', 1],
        ['ALARM2', 1],
        ['ALARM3', 1],
        ['ALARM4', 1],
        ['ALARM5', 1],
        ['ALARM6', 1],
        ['ALARM7', 1],
        ['ALARM8', 1],
        ['ALARM9', 1],
        ['ALARM10', 1],
        ['ALARM11', 1],
        ['ALARM12', 1],
        ['ALARM13', 1],
        ['ALARM14', 1],
        ['ALARM15', 1],
    ]
elif event == 'SCREAM':
    gain = 1
    list_bag_target = [
        ['SCREAM1', 1],
        ['SCREAM2', 1],
        ['SCREAM3', 1],
        ['SCREAM4', 1],
        ['SCREAM5', 1],
        ['SCREAM6', 1],
        ['SCREAM7', 1],
        ['SCREAM8', 1],
        ['SCREAM9', 1],
        ['SCREAM10', 1],
        ['SCREAM11', 1],
        ['SCREAM12', 1],
        ['SCREAM13', 1],
        ['SCREAM14', 1],
        ['SCREAM15', 1],
    ]
elif event=='SPEECH':
    gain = 0.8
    list_bag_target = [
    ['237-126133-0000', 0.8],
    ['237-126133-0001', 0.8],
    ['237-126133-0003', 0.8],
    ['237-126133-0005', 0.8],
    ['237-126133-0023', 0.8],
    ['237-126133-0024', 0.8],

    ['260-123286-0003', 0.8],
    ['260-123286-0006', 0.8],
    ['260-123286-0016', 0.8],
    ['260-123286-0026', 0.8],
    ['260-123286-0027', 0.8],
    ['260-123286-0028', 0.8],

    ['1089-134686-0000', 0.8],
    ['1089-134686-0002', 0.8],
    ['1089-134686-0005', 0.8],
    ['1089-134686-0011', 0.8],
    ['1089-134686-0012', 0.8],
    ['1089-134686-0013', 0.8],
    #
    ['1188-133604-0003', 0.8],
    ['1188-133604-0004', 0.8],
    ['1188-133604-0007', 0.8],
    ['1188-133604-0018', 0.8],
    ['1188-133604-0019', 0.8],
    ['1188-133604-0021', 0.8],
    #
    ['3570-5694-0000', 0.8],
    ['3570-5694-0005', 0.8],
    ['3570-5694-0006', 0.8],
    ['3570-5694-0007', 0.8],
    ['3570-5694-0008', 0.8],
    ['3570-5694-0011', 0.8],
    #
    ['4992-23283-0005', 0.8],
    ['4992-23283-0009', 0.8],
    ['4992-23283-0017', 0.8],
    ['4992-23283-0020', 0.8],
    ['4992-41797-0008', 0.8],
    ['4992-41797-0009', 0.8]
    ]

