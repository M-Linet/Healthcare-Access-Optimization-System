HOW TO ADD YOUR OWN REAL PHOTOS
================================

The app now looks for images in a folder your backend already serves.
It will NOT break if these files are missing (you'll just see a clean
dashed placeholder instead of a broken image icon) -- but for the full
effect, add these 4 files:

1. Create the folder:  frontend/images/

2. Add these exact filenames inside it:
   - hero.jpg         -> wide background photo behind the title (any aspect ratio, will be cropped)
   - dispensary.jpg   -> photo representing a KEPH Level 2 dispensary
   - community.jpg    -> photo representing community health outreach
   - hospital.jpg     -> photo representing a KEPH Level 6 national referral hospital

WHERE TO FIND REAL, FREE-TO-USE KENYAN HEALTHCARE PHOTOS:
   - Wikimedia Commons: https://commons.wikimedia.org/wiki/Category:Healthcare_in_Kenya
     (search this category directly in your browser, right-click > Save Image on any you like)
   - Unsplash: https://unsplash.com/s/photos/kenya-hospital
   - Direct Relief photo library: https://www.directrelief.org (Kenya program photos)

WHY LOCAL FILES INSTEAD OF LINKING TO A WEBSITE:
   Hotlinked images depend on an external site staying online and reachable.
   During a live demo (e.g. your defense), that's a risk you don't need --
   local files always work, load instantly, and you control exactly which
   photos represent your project.

No code changes needed after adding the files -- just refresh the page.
