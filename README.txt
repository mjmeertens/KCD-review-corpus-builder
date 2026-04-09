Hi!

The reviews for the final analysis were scraped on 07/04/2026 15:54. Results may be different if you decide to run the script again. More on this is in ABOUT.txt in the scripts folder.

I have made the entire corpus myself and coded everything. This folder has the whole process.
'scripts' has all the scripts. I would recommend you read ABOUT.txt, where I motivate some choices. You will need to run the scraper and the preprocesser yourself if you wish to run analysis.py without the cache, because they're too big to upload. I'll fix that some other time. 'output' contains output files for sentiment analysis.

Because of the large amounts of data, I have cached the intermediate steps so you can safely run the code without your computer crashing. These are in the 'cache' folder. Make sure you're running the script from the right interpreter and the right working directory so it finds the .pkl files. If not, it will still run the heavy parts. 

Feel free to use the script to build a corpus for game reviews of any other steam game. The analysis script is made specifically to the KCD corpus, but if you have another game corpus, it should be very easy to change the variables to your own. Let me know if you did so, I would be very curious for the results :)

Huge credit to woctezuma on GitHub for the steamreviews code, it works flawlessly: https://github.com/woctezuma/download-steam-reviews.

Happy coding!
Meike
