import argparse
import os
import sys
import warnings
import pandas as pd

warnings.filterwarnings('ignore')
from phishing_detector import PhishingURLDetector

def argument_parsing():
    parser = argparse.ArgumentParser(description='Train and compare Phishing URL Detector models.')
    parser.add_argument("-ep", "--epochs", type=int, default=20,
                        help='The number of training epochs for Keras models.')
    parser.add_argument("-bs", "--batch_size", type=int, default=64,
                        help='Batch size for training Keras models.')
    parser.add_argument("-mr", "--max_rows", type=int, default=50000,
                        help='Maximum rows used from the training data.')
    parser.add_argument("-sf", "--sample_fraction", type=float, default=1.0,
                        help='Fraction of the dataset to sample for training (e.g., 0.1 for 10%%).')
    parser.add_argument("-max_features", "--max_tfidf_features", type=int, default=5000,
                        help='Maximum number of TF-IDF features.')
    parser.add_argument("-model_dir", "--model_directory", type=str, default='saved_models',
                        help='Base directory to save/load all models and preprocessors. '
                             'Subdirectories will be created for each architecture.')
    parser.add_argument("-train_data", "--train_data_path", type=str, 
                        default='data/small_dataset/train.txt',
                        help='Path to the training data file.')
    parser.add_argument("-test_data", "--test_data_path", type=str, 
                        default='data/small_dataset/test.txt',
                        help='Path to the test data file.')
    parser.add_argument("-val_data", "--val_data_path", type=str, 
                        default='data/small_dataset/val.txt',
                        help='Path to the validation data file.')
    
    args = parser.parse_args()
    return args

def main():
    """
    Main function to run the phishing URL detection training and evaluation
    across multiple architectures.
    """
    args = argument_parsing()

    # Initialize the detector with specified max TF-IDF features
    detector = PhishingURLDetector(max_features=args.max_tfidf_features)

    # Define paths for data files
    train_file = args.train_data_path
    test_file = args.test_data_path
    val_file = args.val_data_path

    # Load data once for all models
    df = detector.data_processor.load_data(train_file, test_file, val_file)

    if not df.empty:
        # Architectures to test
        architectures_to_test = ["ANN", "CNN", "XGBoost"]  # removing rnn

        # Calculate sample size based on fraction if specified
        total_samples = len(df)
        sample_size = int(total_samples * args.sample_fraction)
        if args.sample_fraction < 1.0:
            print(f"Sampling {args.sample_fraction*100:.0f}% of data, approximately {sample_size} samples.")

        for arch_name in architectures_to_test:
            try:
                print(f"\n{'='*80}\nStarting training for {arch_name} architecture...\n{'='*80}")
                
                # Train the model for the current architecture
                detector.train_model(
                    df, 
                    architecture=arch_name, 
                    epochs=args.epochs, 
                    batch_size=args.batch_size, 
                    sample_size=sample_size
                )

                # Create architecture-specific save directory for plots
                arch_plot_dir = os.path.join(args.model_directory, arch_name.lower())
                os.makedirs(arch_plot_dir, exist_ok=True)

                # Save the complete pipeline for the current architecture
                detector.save_pipeline(architecture_name=arch_name, model_dir=args.model_directory)

                # Generate and save training plots (only for Keras models)
                if arch_name in ["ANN", "CNN", "RNN"]:
                    print(f"\n📊 Generating training history plots for {arch_name}...")
                    plot_path = os.path.join(args.model_directory, arch_name.lower(), f'{arch_name.lower()}_training_history.png')
                    detector.plot_training_history(architecture_name=arch_name, save_path_base=plot_path.replace('.png', '')) # Remove .png for base
                
                # Generate and save confusion matrix and classification report
                print(f"📈 Generating confusion matrix and performance metrics plots for {arch_name}...")
                plot_path = os.path.join(args.model_directory, arch_name.lower(), f'{arch_name.lower()}_performance_metrics.png')
                    # Pass the directory path for saving plots
                detector.generate_metrics_plots(architecture_name=arch_name, save_path_base=arch_plot_dir)

            except KeyboardInterrupt:
                print(f"\nTraining for {arch_name} interrupted by user!")
                break 
            except Exception as e:
                print(f"\nFATAL UNHANDLED ERROR DURING {arch_name} EXECUTION: {e}")
                import traceback
                traceback.print_exc()
                # Continue to next architecture even if one fails
        
        # After training all models, plot the comparison
        if detector.all_models_performance:
            print(f"\n{'='*80}\nGenerating overall model comparison plot...\n{'='*80}")
            comparison_plot_path = os.path.join(args.model_directory, 'model_comparison_summary.png')
            detector.plot_all_models_comparison(save_path=comparison_plot_path)
            print(f"✅ All model comparison plot saved to {comparison_plot_path}")
        else:
            print("\nNo models were successfully trained to generate a comparison plot.")


        print("\n🔮 Testing predictions on example URLs using the pipelines (requires `inference.py` script):")
        print("\nTo test saved pipelines, run `python inference.py` and specify the model architecture.")
        print("Example URLs for manual testing:")
        test_urls = [
            "http://www.bartekbitner.pl/libraries/fof/-/din7",
            "https://eheadspace.org.au/headspace-centres/murray-bridge/","https://www.google.com",
            "http://malicious-site-12345.com/fake-login","http://facebook.com.phishing.site.biz/login",
            "https://www.wellsfargo.com","http://tinyurl.com/some-phishing-link",
            "https://www.amazon.co.jp.safe-login.com/signin"
        ]
        for url in test_urls:
            print(f"- {url}")

    else:
        print("❌ No data loaded or data frame is empty. Please check your data files and their content.")

if __name__ == '__main__':
    main()
