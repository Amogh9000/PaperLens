# main.py - Main OMR System Script
"""
Enhanced OMR System with Heading-Based Detection
- Detects subject headings to find answer area
- Crops out top part (headings)
- Aligns bubbles into 4 columns (A, B, C, D)
- Uses cleaned image for better detection
"""

from engine import flatten_image, OMRGrader
import json
import cv2
import os

def main():
    """
    Main function to run the OMR system
    """
    print("=" * 60)
    print("🎯 ENHANCED OMR SYSTEM")
    print("=" * 60)
    print("Features:")
    print("✅ Heading-based answer area detection")
    print("✅ Cropped image (no top part)")
    print("✅ 4-column bubble alignment (A, B, C, D)")
    print("✅ Number filtering for better accuracy")
    print("✅ Cleaned image processing")
    print("=" * 60)
    
    # Configuration - Change these as needed
    IMAGE_PATH = 'Img2.jpeg'  # Change to your image file
    ANSWER_KEY_PATH = 'answer_key.json'
    SET_NAME = 'set_1'  # Change to 'set_2' if using set 2
    
    # Check if files exist
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ ERROR: Image file not found: {IMAGE_PATH}")
        print("   Available images: Img1.jpeg, Img2.jpeg, Img20.jpeg")
        return
    
    if not os.path.exists(ANSWER_KEY_PATH):
        print(f"❌ ERROR: Answer key file not found: {ANSWER_KEY_PATH}")
        return
    
    try:
        # Step 1: Flatten the image
        print(f"\n🔄 STEP 1: Processing Image")
        print("-" * 30)
        print(f"   Image: {IMAGE_PATH}")
        flattened = flatten_image(IMAGE_PATH, show_steps=False)
        print("✅ Image flattened and perspective corrected!")
        print(f"   Shape: {flattened.shape}")
        
        # Step 2: Load answer key
        print(f"\n🔄 STEP 2: Loading Answer Key")
        print("-" * 30)
        with open(ANSWER_KEY_PATH, 'r') as f:
            answer_key_data = json.load(f)
        print(f"✅ Answer key loaded for {SET_NAME}")
        print(f"   Questions: {len(answer_key_data[SET_NAME])}")
        
        # Step 3: Process OMR
        print(f"\n🔄 STEP 3: OMR Processing")
        print("-" * 30)
        print("   - Detecting subject headings")
        print("   - Cropping answer area")
        print("   - Aligning bubbles into 4 columns")
        print("   - Detecting filled answers")
        
        grader = OMRGrader(answer_key_data)
        results = grader.grade(flattened, set_name=SET_NAME, debug=False)
        
        # Step 4: Display results
        print(f"\n" + "=" * 60)
        print("📊 RESULTS")
        print("=" * 60)
        
        print(f"🎯 Overall Performance:")
        print(f"   Score: {results['total_score']} / {results['total_questions']}")
        print(f"   Accuracy: {results['accuracy']:.2f}%")
        
        print(f"\n📚 Subject-wise Scores:")
        for subject, score in results['subject_scores'].items():
            subject_range = grader.SUBJECTS[subject]
            max_possible = subject_range[1] - subject_range[0] + 1
            percentage = (score / max_possible) * 100 if max_possible > 0 else 0
            print(f"   {subject}: {score}/{max_possible} ({percentage:.1f}%)")
        
        # Show first 20 detailed results
        print(f"\n📝 First 20 Questions:")
        print("-" * 40)
        for i, result in enumerate(results['detailed_results'][:20]):
            status = "✅" if result['is_correct'] else "❌"
            print(f"   Q{result['question']}: {result['student_answer']} "
                  f"(Correct: {result['correct_answer']}) {status}")
        
        # Analysis
        correct_count = sum(1 for r in results['detailed_results'] if r['is_correct'])
        incorrect_count = sum(1 for r in results['detailed_results'] 
                            if not r['is_correct'] and r['student_answer'] != 'No Answer')
        unanswered_count = sum(1 for r in results['detailed_results'] 
                             if r['student_answer'] == 'No Answer')
        
        print(f"\n📈 Summary:")
        print(f"   ✅ Correct: {correct_count}")
        print(f"   ❌ Incorrect: {incorrect_count}")
        print(f"   ❓ Unanswered: {unanswered_count}")
        
        # Save results
        results_file = f"omr_results_{SET_NAME}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {results_file}")
        
        print(f"\n" + "=" * 60)
        print("✅ OMR PROCESSING COMPLETED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
