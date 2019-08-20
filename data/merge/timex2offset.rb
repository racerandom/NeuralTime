prevtag = "" 
File.open(ARGV[0],"r:utf-8").each do |line|
  line.chomp!
  array = line.scan(/<.*?>|./)
  array.delete(" ")

  array.each do |token|
    if token !~ /^</
      puts token + "\t" + prevtag
      prevtag = ""
    elsif token =~ /^<(TIMEX3|EVENT)/
      prevtag = token unless token =~/CREATION_TIME/
      token.gsub!(/(<|>)/,"")
    end
  end
end
