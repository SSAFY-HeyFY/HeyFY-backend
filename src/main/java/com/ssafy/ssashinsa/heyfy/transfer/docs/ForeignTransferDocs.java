package com.ssafy.ssashinsa.heyfy.transfer.docs;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import com.ssafy.ssashinsa.heyfy.transfer.dto.TransferHistoryResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Operation(summary = "외환 계좌 이체", description = "계좌로 금액을 이체합니다.")
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "성공적으로 이체를 완료했습니다.",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = TransferHistoryResponse.class),
                        examples = @ExampleObject(
                                name = "계좌 이체 성공 예시",
                                value = "{\n  \"success\": true,\n  \"history\": {\n    \"fromAccountMasked\": \"string\",\n    \"toAccountMasked\": \"string\",\n    \"amount\": \"string\",\n    \"currency\": \"USD\",\n    \"transactionSummary\": \"string\",\n    \"completedAt\": \"2025-08-23T23:13:49.593964+09:00\"\n  },\n  \"error\": null\n}"
                        )
                )
        ),
        @ApiResponse(
                responseCode = "400",
                description = "잘못된 요청: 출금 계좌를 찾을 수 없음",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = @ExampleObject(
                                name = "출금 계좌 없음",
                                summary = "현재 유저에게 연결된 출금 가능한 계좌가 없는 경우",
                                value = "{\"status\": 400, \"httpError\": \"BAD_REQUEST\", \"errorCode\": \"WITHDRAWAL_ACCOUNT_NOT_FOUND\", \"message\": \"[A1003] 계좌번호가 유효하지 않습니다.\"}"
                        )
                )
        ),
        @ApiResponse(
                responseCode = "500",
                description = "서버 내부 오류 또는 API 호출 실패",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = @ExampleObject(
                                name = "API 호출 실패",
                                summary = "신한은행 이체 API 통신 중 오류가 발생한 경우",
                                value = "{\n  \"status\": 500,\n  \"httpError\": \"INTERNAL_SERVER_ERROR\",\n  \"errorCode\": \"API_CALL_FAILED\",\n  \"message\": \"신한 API 호출에 실패했습니다.\"\n}"
                        )
                )
        )
})
public @interface ForeignTransferDocs {
}