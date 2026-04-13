"""
MediaPipe Pose Detector 테스트 스크립트
"""

import cv2
import json
from app.services.mediapipe_detector import MediaPipePoseDetector


def test_static_image(image_path: str):
    """정적 이미지에서 포즈 감지 테스트"""
    detector = MediaPipePoseDetector()

    # 이미지 로드
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
        return

    print(f"✅ 이미지 로드 성공: {image_path}")
    print(f"   크기: {frame.shape[1]}x{frame.shape[0]}")

    # 포즈 감지
    result = detector.detect_pose(frame)

    # 결과 출력
    print(f"\n📊 포즈 감지 결과:")
    print(f"   감지됨: {result.is_detected}")
    print(f"   신뢰도: {result.confidence:.4f}")
    print(f"   감지된 포인트: {len(result.landmarks)}")

    if result.is_detected:
        print(f"\n🔍 상위 5개 포인트 (신뢰도 순):")
        sorted_landmarks = sorted(result.landmarks, key=lambda x: x.visibility, reverse=True)
        for lm in sorted_landmarks[:5]:
            print(f"   {lm.id:2d}. {lm.name:15s} | x:{lm.x:.3f} y:{lm.y:.3f} | 신뢰도:{lm.visibility:.3f}")

    # 결과를 이미지에 그리기
    annotated_frame = detector.draw_landmarks_on_frame(frame, result)

    # 결과 저장
    output_path = image_path.replace(".jpg", "_result.jpg").replace(".png", "_result.png")
    cv2.imwrite(output_path, annotated_frame)
    print(f"\n💾 결과 이미지 저장: {output_path}")

    # JSON 출력
    print(f"\n📋 JSON 형식 결과:")
    result_dict = detector.to_dict(result)
    print(json.dumps(result_dict, indent=2, ensure_ascii=False))

    detector.release()


def test_webcam_simple():
    """웹캠 테스트 (간단 버전)"""
    detector = MediaPipePoseDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다.")
        return

    print("✅ 웹캠 시작 (30프레임 또는 ESC로 종료)")
    print("-" * 50)

    frame_count = 0
    detected_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect_pose(frame)
        annotated_frame = detector.draw_landmarks_on_frame(frame, result)

        # 신뢰도 표시
        if result.is_detected:
            text = f"Detected | Conf: {result.confidence:.2f}"
            color = (0, 255, 0)
            detected_count += 1
        else:
            text = "No Pose"
            color = (0, 0, 255)

        cv2.putText(annotated_frame, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Test - Press ESC to stop", annotated_frame)

        frame_count += 1

        # 30프레임 또는 ESC로 종료
        if frame_count >= 30 or cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n📊 테스트 완료:")
    print(f"   총 프레임: {frame_count}")
    print(f"   감지됨: {detected_count} ({detected_count/frame_count*100:.1f}%)")

    detector.release()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 이미지 경로가 주어진 경우
        image_path = sys.argv[1]
        test_static_image(image_path)
    else:
        # 웹캠 테스트
        test_webcam_simple()