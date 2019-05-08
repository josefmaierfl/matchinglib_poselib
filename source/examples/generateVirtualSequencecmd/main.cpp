
 #include "loadMatches.h"
 #include "argvparser.h"

 using namespace CommandLineProcessing;

 void SetupCommandlineParser(ArgvParser& cmd, int argc, char* argv[])
 {
     cmd.setIntroductoryDescription("Example program to show how the generated matches should be loaded.");
     //define error codes
     cmd.addErrorCode(0, "Success");
     cmd.addErrorCode(1, "Error");

     cmd.setHelpOption("h", "help","Using option --file is mandatory.");
     cmd.defineOption("file", "<Path and Filename (including file ending) of the matches.>",
             ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);

     /// finally parse and handle return codes (display help etc...)
     int result = -1;
     result = cmd.parse(argc, argv);
     if (result != ArgvParser::NoParserError)
     {
         cout << cmd.parseErrorDescription(result);
         exit(1);
     }
 }

 int main(int argc, char* argv[])
 {
     ArgvParser cmd;
     SetupCommandlineParser(cmd, argc, argv);

     std::string filename = cmd.optionValue("file");
     sequMatches sm;
     if(!readMatchesFromDisk(filename, sm)){
         cerr << "Unable to load matches." << endl;
         return EXIT_FAILURE;
     }else{
         cout << "Loading matches successful" << endl;
     }

     return EXIT_SUCCESS;
 }
