linenum = 0
File.open(ARGV[0],"r:utf-8").each do |line|
  line.chomp!
  linenum += 1
  next if line =~ /^#/
  next if line =~ /^$/
  array = line.split("\t")
  chars = array[1].split(//)
  chars.each do |char|
   puts char + "\t" + linenum.to_s
  end
end
