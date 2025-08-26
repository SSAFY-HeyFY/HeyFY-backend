package com.ssafy.ssashinsa.heyfy.fcm.docs;

import com.ssafy.ssashinsa.heyfy.fcm.dto.FcmTokenRequest;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.parameters.RequestBody;
import io.swagger.v3.oas.annotations.responses.*;

import java.lang.annotation.*;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Operation(
        summary = "FCM 토큰 공개 삭제 (로그인 불필요)",
        description = "RefreshToken 만료 등 인증이 불가한 상황에서 클라이언트가 기기 FCM 토큰을 서버 DB에서 삭제합니다."
)
@ApiResponses({
        @ApiResponse(responseCode = "200", description = "삭제 성공"),
        @ApiResponse(responseCode = "400", description = "필수 정보 누락"),
        @ApiResponse(responseCode = "500", description = "서버 에러")
})
@RequestBody(
        description = "삭제할 기기의 FCM 토큰",
        required = true,
        content = @Content(
                schema = @Schema(implementation = FcmTokenRequest.class),
                examples = @ExampleObject(value = "{ \"fcmToken\": \"fH7Zc...abcd1234\" }")
        )
)
public @interface FcmDeletePublicDocs {}
