# coding: cp932
# text = �ӂ��ƂՂ炴
File.open(ARGV[0]).each_line do |line|
  if line =~ /^# text = /
    puts line.sub( /^# text = /,"")
  end
end
