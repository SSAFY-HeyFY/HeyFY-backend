package com.ssafy.ssashinsa.heyfy.fcm.docs;

import com.ssafy.ssashinsa.heyfy.fcm.dto.NotificationRequest;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.parameters.RequestBody;
import io.swagger.v3.oas.annotations.responses.*;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.tags.Tag;

import java.lang.annotation.*;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Operation(
        summary = "모든 토큰으로 푸시 발송",
        description = "모든 FCM 토큰으로 알림을 보내는 테스트/QA용 API입니다."
)
@ApiResponses({
        @ApiResponse(responseCode = "200", description = "전송 성공"),
        @ApiResponse(
                responseCode = "500",
                description = "서버 에러",
                content = @Content(
                        mediaType = "application/json",
                        examples = @ExampleObject(ref = "#/components/examples/InternalError")
                )
        )
})
@RequestBody(
        description = "발송 대상 및 내용",
        required = true,
        content = @Content(
                schema = @Schema(implementation = NotificationRequest.class),
                examples = @ExampleObject(
                        name = "발송 예시",
                        value = "{ \"title\": \"공지\", \"body\": \"점검 안내\" }"
                )
        )
)
public @interface NotificationAllSendDocs {}
